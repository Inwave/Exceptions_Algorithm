import cv2
import yaml
import numpy as np
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from frame_capture import FrameDecoder
from detection_analysis import Detector
from profiler import profiled

DEFAULT_COLORS = {
    'scanner':(255, 0, 0), 'hand': (0, 255, 0), 'rag' :(0, 0, 255),
    'plastic-bag':(255, 255, 0), 'products':(0, 255, 255), 'price-sheets':(255, 0, 255)
}

class UnifiedWatcher:
    def __init__(self, config_path: str, detector: Any, video_path: str):
        self.video_path = video_path
        self.detector = detector

        # --- Charger la config ---
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # PersistentObjectWatcher config
        pw_cfg = config.get('persistentwatcher', {})
        self.pw_roi = pw_cfg.get('roi', None)
        self.pw_time_threshold = pw_cfg.get('time_threshold', 5.0)
        self.pw_smoothing_window = pw_cfg.get('smoothing_window', 10)
        self.pw_alpha = pw_cfg.get('alpha', 0.8)
        self.pw_motion_threshold = pw_cfg.get('motion_threshold', 0.01)
        self.pw_disappearance_window = pw_cfg.get('disappearance_window', 10)
        self.pw_target_classes = pw_cfg.get('target_classes', [])
        self.display_scale = pw_cfg.get('display_scale', 1.0)
        self.pw_background_frame_index= pw_cfg.get('background_frame_index', 0)
        self.pw_window_size = pw_cfg.get('window_size', 7)
        self.pw_min_low_frames = pw_cfg.get('min_low_frames', 3)
        self.pw_high_similarity_threshold = pw_cfg.get('high_similarity_threshold', 0.8)
        self.pw_low_similarity_threshold = pw_cfg.get('low_similarity_threshold', 0.5)
        self.pw_min_immobility_duration = pw_cfg.get('min_immobility_duration', 3)
        self.pw_min_high_frames = pw_cfg.get('min_high_frames', 3)


        # SimpleTracking config
        st_cfg = config.get('simpletracking', {})
        self.st_roi = st_cfg.get('roi', None)
        self.st_target_classes = st_cfg.get('target_classes', [])
        self.st_frame_skip = st_cfg.get('frame_skip', 0)
        self.st_resize = None
        self.st_movement_threshold = st_cfg.get('movement_threshold', 0.015)
        self.st_smoothing_window = st_cfg.get('smoothing_window', 25)
        self.st_alpha = st_cfg.get('alpha', 0.8)
        self.st_min_movement_count = st_cfg.get('min_movement_count', 5)
        self.st_movement_window = st_cfg.get('movement_window', 12)
        self.st_movement_display_timer = {cls: 0 for cls in self.st_target_classes}

        # --- Internal states ---
        self.pw_prev_mask = None
        self.pw_similarity_window = deque(maxlen=7)
        self.pw_immobile = False
        self.pw_immobile_start_frame = None
        self.pw_immobile_start_timestamp = None
        self.pw_sequences = []
        
        self.st_positions = {cls: deque(maxlen=self.st_smoothing_window) for cls in self.st_target_classes}
        self.st_deviation_flags = {cls: deque(maxlen=self.st_movement_window) for cls in self.st_target_classes}
        self.st_tracking_data = {cls: [] for cls in self.st_target_classes}
        self.st_class_colors = DEFAULT_COLORS
        self.st_alert_messages = {cls: f"MOVEMENT DETECTED: {cls.upper()}!" for cls in self.st_target_classes}

        # Video resolution
        self.width, self.height = self._get_video_resolution()
        self.alert_messages = {
            cls: f"MOVEMENT DETECTED: {cls.upper()}!" for cls in self.st_target_classes
        }
        self.class_colors = DEFAULT_COLORS


    def _get_video_resolution(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (width, height)

    # ---------------------- Utilitaires ----------------------
    def _bbox_center(self, bbox: List[int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _smooth_position(self, cls: str, new_pos) -> Tuple[float, float]:
        if not self.st_positions[cls]:
            self.st_positions[cls].append(new_pos)
            return new_pos
        last = self.st_positions[cls][-1]
        smoothed = (self.st_alpha * new_pos[0] + (1 - self.st_alpha) * last[0],
                    self.st_alpha * new_pos[1] + (1 - self.st_alpha) * last[1])
        self.st_positions[cls].append(smoothed)
        return smoothed

    def _detect_movement_pattern(self, cls: str, center: Tuple[float, float]) -> bool:
        if len(self.st_positions[cls]) < 2:
            return False
        mean_x = np.mean([p[0] for p in self.st_positions[cls]])
        mean_y = np.mean([p[1] for p in self.st_positions[cls]])
        dist = np.linalg.norm(np.array(center) - np.array([mean_x, mean_y]))
        diag = np.sqrt(self.width**2 + self.height**2)
        norm_dist = dist / diag
        self.st_deviation_flags[cls].append(norm_dist > self.st_movement_threshold)
        return sum(self.st_deviation_flags[cls]) >= self.st_min_movement_count

    @staticmethod
    def compute_mask_similarity(fg_mask, prev_mask, min_area_ratio=0.15):
        total_area = fg_mask.shape[0] * fg_mask.shape[1]
        min_pixels = int(total_area * min_area_ratio)
        intersection = cv2.bitwise_and(fg_mask, prev_mask)
        union = cv2.bitwise_or(fg_mask, prev_mask)
        intersection_area = cv2.countNonZero(intersection)
        union_area = cv2.countNonZero(union)
        if union_area < min_pixels:
            return 0.0
        return intersection_area / union_area

    def _is_inside_roi(self, bbox: List[int], roi: Optional[List[List[int]]]) -> bool:
        if roi is None:
            return True
        x1, y1, x2, y2 = bbox
        for r in roi:
            rx1, ry1, rx2, ry2 = r
            if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):
                return True
        return False
    


    # ---------------------- Detection on frames ----------------------
    def run(self, show_video: bool = False, save_path: Optional[str] = None):
        print(self.display_scale)
        decoder = FrameDecoder(self.video_path, frame_skip=self.st_frame_skip, resize=self.st_resize)
        fps = decoder.fps or 16

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(self.width * max(self.display_scale, self.display_scale))
            height = int(self.height * max(self.display_scale, self.display_scale))
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        movement_info = {
            cls: {
                "detected": False,
                "movement_count": 0,
                "movements": [],
                "_active": False,  # indicate if there's an ongoing movement
                "_start_frame": None,
                "_start_time": None
            }
            for cls in self.st_target_classes
        }

        # --- Background reference frame pour immobility ---
        background_frame = None
        for idx, _, frame in decoder.frames():
            if idx == self.pw_background_frame_index:
                background_frame = frame.copy()
                break
        
        x1, y1, x2, y2 = self.pw_roi if self.pw_roi else (0,0,background_frame.shape[1], background_frame.shape[0])
        bg_roi = background_frame[y1:y2, x1:x2]

        # Detection state
        prev_mask = None
        similarity_window = deque(maxlen=self.pw_window_size)
        
        last_timestamp = 0.0

        # --- Iteration on frames ---
        for idx, timestamp, frame in decoder.frames():
            vis_frame1 = frame.copy()
            vis_frame2=frame.copy()
            last_timestamp = timestamp

            # --- PersistentObjectWatcher immobility ---
            roi_frame = vis_frame2[y1:y2, x1:x2]
            diff = cv2.absdiff(roi_frame, bg_roi)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            gray_diff = cv2.GaussianBlur(gray_diff, (5,5),0)
            _, fg_mask = cv2.threshold(gray_diff,30,255,cv2.THRESH_BINARY)
            sim = self.compute_mask_similarity(fg_mask, self.pw_prev_mask, min_area_ratio=0.1) if self.pw_prev_mask is not None else 0.0
            self.pw_prev_mask = fg_mask.copy()
            self.pw_similarity_window.append(sim)

            if len(self.pw_similarity_window) >= 7:
                high_count = sum(1 for s in self.pw_similarity_window if s >= self.pw_high_similarity_threshold)
                low_count = sum(1 for s in self.pw_similarity_window if s < self.pw_low_similarity_threshold)

                if not self.pw_immobile and high_count >= self.pw_min_high_frames:
                    self.pw_immobile = True
                    self.pw_immobile_start_frame = idx - 6
                    self.pw_immobile_start_timestamp = timestamp - 6 / fps

                elif self.pw_immobile and low_count >= self.pw_min_low_frames:
                    self.pw_immobile = False
                    duration = timestamp - self.pw_immobile_start_timestamp
                    if duration >= self.pw_min_immobility_duration:
                        self.pw_sequences.append({'start_frame': self.pw_immobile_start_frame,
                                                  'end_frame': idx,
                                                  'start_timestamp': self.pw_immobile_start_timestamp,
                                                  'end_timestamp': timestamp,
                                                  'duration': duration})

                    self.pw_immobile_start_frame = None
                    self.pw_immobile_start_timestamp = None

            # --- SimpleTracking ---
            detections = self.detector.detect_frame(frame)
            for cls in self.st_target_classes:
                target_dets = [d for d in detections if d['class_name']==cls and self._is_inside_roi(d['bbox'], self.st_roi)]
                if not target_dets:
                    continue
                
                target = max(target_dets, key=lambda d: d['conf'])
                bbox = target['bbox']
                center = self._bbox_center(bbox)
                smooth_center = self._smooth_position(cls, center)
                self.st_tracking_data[cls].append({"frame_idx": idx, "timestamp": timestamp, "bbox": bbox, "center": smooth_center})
                moving = self._detect_movement_pattern(cls, smooth_center)
                cls_state = movement_info[cls]

                if moving and not cls_state['_active']:
                    cls_state['_active'] = True
                    cls_state['_start_frame'] = idx
                    cls_state['_start_time'] = timestamp
                elif not moving and cls_state['_active']:
                    cls_state['_active'] = False
                    cls_state['movements'].append({"start_frame": cls_state['_start_frame'], "end_frame": idx, "start_time": cls_state['_start_time'], "end_time": timestamp})
                    cls_state["movement_count"] += 1
                    cls_state["detected"] = True
                if cls_state['_active']:
                    msg = self.alert_messages[cls]
                    if show_video:
                        cv2.putText(
                    vis_frame1, msg, (30, 50 + 40 * list(self.st_target_classes).index(cls)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.class_colors[cls], 3, cv2.LINE_AA
                )
                # Alert text
                if self.st_movement_display_timer[cls] > 0:
                    msg = self.alert_messages[cls]
                    cv2.putText(
                        vis_frame1, msg, (30, 50 + 40 * list(self.st_target_classes).index(cls)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.class_colors[cls], 3, cv2.LINE_AA
                    )
                    self.st_movement_display_timer[cls] -= 1


            # --- Visualization ---
                if show_video:

                    # Tracker visualization:

                    cv2.rectangle(vis_frame1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.class_colors[cls], 2)
                    cx, cy = map(int, smooth_center)
                    cv2.circle(vis_frame1, (cx, cy), 5, self.class_colors[cls], -1)

                    # Trajectory
                    pts = self.st_positions[cls]
                    for i in range(1, len(pts)):
                        p1 = tuple(map(int, pts[i - 1]))
                        p2 = tuple(map(int, pts[i]))
                        cv2.line(vis_frame1, p1, p2, self.class_colors[cls], 2)

                    

                # Persistent object watcher visualization:
            if show_video:
                color = (0, 0, 255) if self.pw_immobile else (0, 255, 0)
                cv2.rectangle(vis_frame2, (x1, y1), (x2, y2), color, 2)
                status = "Immobile" if self.pw_immobile else "Mobile"
                cv2.putText(vis_frame2, f"State: {status}", (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(vis_frame2, f"Sim: {sim:.2f}", (x1 + 10, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                

                

            # Draw ROI(s)
                if self.st_roi:
                    for roi in self.st_roi:
                        cv2.rectangle(vis_frame1, roi[:2], roi[2:], (100, 255, 100), 2)
                if self.pw_roi:
                    cv2.rectangle(vis_frame2, self.pw_roi[:2], self.pw_roi[2:], (100, 255, 100), 2)
                # Resize + Display
                if self.display_scale != 1.0:
                    vis_frame1 = cv2.resize(vis_frame1, (0, 0), fx=self.display_scale, fy=self.display_scale)
                    vis_frame2 = cv2.resize(vis_frame2, (0, 0), fx=self.display_scale, fy=self.display_scale)
            
                combined_display = np.hstack((vis_frame1, vis_frame2))

                cv2.imshow("Detections", combined_display)
                key = cv2.waitKey(int(1000 / fps)) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space bar: pause and step through frames
                    paused = True
                    while paused:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord(' '):  # Space: next frame
                            paused = False
                        elif key == ord('q'):  # Q: quit
                            break
                    if key == ord('q'):
                        break

        #Close ongoing immobility at video end:
        if self.pw_immobile and self.pw_immobile_start_frame is not None:
            duration = last_timestamp - self.pw_immobile_start_timestamp
            if duration >= self.pw_min_immobility_duration:
                self.pw_sequences.append({
                    'start_frame': self.pw_immobile_start_frame,
                    'end_frame': idx,
                    'start_timestamp': self.pw_immobile_start_timestamp,
                    'end_timestamp': last_timestamp,
                    'duration': duration
                })

            self.pw_immobile_start_frame = None
            self.pw_immobile_start_timestamp = None
            self.pw_immobile = False

        # Close active movements
        for cls, cls_state in movement_info.items():
            if cls_state["_active"]:
                last_frame = self.st_tracking_data[cls][-1]["frame_idx"]
                last_time = self.st_tracking_data[cls][-1]["timestamp"]
                start_frame = cls_state["_start_frame"]
                start_time = cls_state["_start_time"]
                duration_s = last_time - start_time
                frame_count = last_frame - start_frame + 1
                cls_state["movements"].append({
                    "start_frame": start_frame,
                    "end_frame": last_frame,
                    "start_timestamp": start_time,
                    "end_timestamp": last_time,
                    "duration_s": duration_s,
                    "frame_count": frame_count
                })
                cls_state["movement_count"] += 1
                cls_state["detected"] = True

            # Deletion of internal variables
            del cls_state["_active"]
            del cls_state["_start_frame"]
            del cls_state["_start_time"]


        decoder.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        return {"persistentwatcher": self.pw_sequences, "simpletracking": movement_info}


if __name__ == "__main__":
    detector=Detector("config.yaml")
    unified_watcher = UnifiedWatcher("config.yaml", detector=detector, video_path="videos/video_test_CarrefourSP_TBE1_20250924T155916_2.mp4")
    results=unified_watcher.run(show_video=True)
    print(results)