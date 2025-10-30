import cv2
import yaml
import os
import shutil
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from frame_capture import FrameDecoder
from detection_analysis import Detector
from profiler import profiled, plot_profile_stats



# Class regrouping the tracking algorithm and the immobility-detection algorithm
# Run simultaneously both on each frame
class UnifiedWatcher:
    @profiled
    def __init__(self, config_path: str, detector: Any):
        self.detector = detector

        # load config from config.yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.img_size = config.get('img_size', 640) # Image size, should be the same as the one used for training for better results.
        self.class_colors = config.get('class_colors', {})

        # PersistentObjectWatcher config
        pw_cfg = config.get('persistentwatcher', {})
        self.background_frame = cv2.imread(pw_cfg.get('background_frame_path')) if pw_cfg.get('background_frame_path') else None
        self.pw_roi = self.denormalize_roi(pw_cfg.get('roi', None))
        self.pw_time_threshold = pw_cfg.get('time_threshold', 1.0)
        self.pw_smoothing_window = pw_cfg.get('smoothing_window', 7)
        self.pw_alpha = pw_cfg.get('alpha', 0.8)
        self.pw_motion_threshold = pw_cfg.get('motion_threshold', 0.1)
        self.pw_disappearance_window = pw_cfg.get('disappearance_window', 10)
        self.pw_target_classes = pw_cfg.get('target_classes', [])
        self.display_scale = pw_cfg.get('display_scale', 1.0)
        self.pw_background_frame_index= pw_cfg.get('background_frame_index', 0)
        self.pw_window_size = pw_cfg.get('window_size', 7)
        self.pw_min_low_frames = pw_cfg.get('min_low_frames', 3)
        self.pw_high_similarity_threshold = pw_cfg.get('high_similarity_threshold', 0.8)
        self.pw_low_similarity_threshold = pw_cfg.get('low_similarity_threshold', 0.5)
        self.pw_min_immobility_duration = pw_cfg.get('min_immobility_duration', 1.5)
        self.pw_min_high_frames = pw_cfg.get('min_high_frames', 6)
        self.min_area_ratio= pw_cfg.get('min_area_ratio', 0.1)
        self.pw_min_class_count = pw_cfg.get('min_class_count', 4)
        self.pw_roi_threshold = pw_cfg.get('roi_threshold', 0.3)
        self.pw_sequence_just_ended = False


        # SimpleTracking config
        st_cfg = config.get('simpletracking', {})
        self.st_roi = self.denormalize_rois(st_cfg.get('roi', None))
        self.st_target_classes = st_cfg.get('target_classes', [])
        self.st_frame_skip = st_cfg.get('frame_skip', 0)
        self.st_resize = None
        self.st_movement_threshold = st_cfg.get('movement_threshold', 0.015)
        self.st_smoothing_window = st_cfg.get('smoothing_window', 25)
        self.st_alpha = st_cfg.get('alpha', 0.8)
        self.st_min_movement_count = st_cfg.get('min_movement_count', 5)
        self.st_movement_window = st_cfg.get('movement_window', 12)
        self.st_movement_display_timer = {cls: 0 for cls in self.st_target_classes}

        # --- Internal states (should be re initialized after each frames sequence) ---
        self.pw_prev_mask = None
        self.pw_similarity_window = deque(maxlen=7)
        self.pw_immobile = False
        self.pw_immobile_start_frame = None
        self.pw_immobile_start_timestamp = None
        self.pw_sequences = []
        
        self.st_positions = {cls: deque(maxlen=self.st_smoothing_window) for cls in self.st_target_classes}
        self.st_valid_positions = {cls: [] for cls in self.st_target_classes}
        self.st_deviation_flags = {cls: deque(maxlen=self.st_movement_window) for cls in self.st_target_classes}
        self.st_tracking_data = {cls: [] for cls in self.st_target_classes}
        self.st_alert_messages = {cls: f"MOVEMENT DETECTED: {cls.upper()}!" for cls in self.st_target_classes}
        
        self.alert_messages = {
            cls: f"MOVEMENT DETECTED: {cls.upper()}!" for cls in self.st_target_classes
        }

    @profiled
    def _get_video_resolution(self, video_path:str=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (width, height)

    # ---------------------- Utils ----------------------
    def _bbox_center(self, bbox: List[int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    # Process and store the last positions of the detected objects
    def _smooth_position(self, cls: str, new_pos: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if new_pos is None:
            self.st_positions[cls].append(None)
            return None

        if not self.st_positions[cls]:
            self.st_positions[cls].append(new_pos)
            return new_pos

        last = self.st_positions[cls][-1]
        if last is None:
            smoothed = new_pos
        else:
            smoothed = (
                self.st_alpha * new_pos[0] + (1 - self.st_alpha) * last[0],
                self.st_alpha * new_pos[1] + (1 - self.st_alpha) * last[1]
            )
        self.st_positions[cls].append(smoothed)
        return smoothed

    # Detect if an object of a certain class is moving significantly
    def _detect_movement_pattern(self, cls: str, center: Tuple[float, float]) -> bool:
        if len(self.st_valid_positions[cls]) < 2:
            return False
        mean_x = np.mean([p[0] for p in self.st_valid_positions[cls]])
        mean_y = np.mean([p[1] for p in self.st_valid_positions[cls]])
        if center is None:
            self.st_deviation_flags[cls].append(False)
            return sum(self.st_deviation_flags[cls]) >= self.st_min_movement_count
        dist = np.linalg.norm(np.array(center) - np.array([mean_x, mean_y]))
        diag = np.sqrt(self.width**2 + self.height**2)
        norm_dist = dist / diag
        self.st_deviation_flags[cls].append(norm_dist > self.st_movement_threshold)
        return sum(self.st_deviation_flags[cls]) >= self.st_min_movement_count


    # Compute the similarity between two masks, allowing to detect if there is an immobility sequence
    def compute_mask_similarity(self,fg_mask, prev_mask):
        total_area = fg_mask.shape[0] * fg_mask.shape[1]
        min_pixels = int(total_area * self.min_area_ratio)
        intersection = cv2.bitwise_and(fg_mask, prev_mask)
        union = cv2.bitwise_or(fg_mask, prev_mask)
        intersection_area = cv2.countNonZero(intersection)
        union_area = cv2.countNonZero(union)
        if union_area < min_pixels:
            return 0.0
        return intersection_area / union_area
    
    # Check if a bbox is inside a single roi
    def _is_inside_roi(self, bbox: List[int], roi:Optional[List[int]]) -> bool:
        if roi is None:
            return True
        x1, y1, x2, y2 = bbox
        rx1, ry1, rx2, ry2 = roi
        if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):
            return True
        return False

# Check if a bbox is significantly inside a single roi
    def _is_significantly_inside_roi(self, bbox: List[int], roi:Optional[List[int]]) -> bool:
        if roi is None:
            return True
        x1, y1, x2, y2 = bbox
        rx1, ry1, rx2, ry2 = roi

        inter_x1 = max(x1, rx1)
        inter_y1 = max(y1, ry1)
        inter_x2 = min(x2, rx2)
        inter_y2 = min(y2, ry2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        bbox_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        roi_area = max(0, (rx2 - rx1)) * max(0, (ry2 - ry1))

        if bbox_area == 0 or roi_area == 0:
            return False

        min_area = min(bbox_area, roi_area)
        ratio = inter_area / min_area

        return ratio > self.pw_roi_threshold
    
    #Check if a bbox is inside an union of rois
    def _is_inside_rois(self, bbox: List[int], roi:Optional[List[List[int]]]) -> bool:
        if roi is None:
            return True
        x1, y1, x2, y2 = bbox
    
        for r in roi:
            rx1, ry1, rx2, ry2 = r
            if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):
                return True
        return False
    
    def denormalize_roi(self, roi_normalized):
        if roi_normalized is None:
            return None
        x1 = int(roi_normalized[0] * self.img_size)
        y1 = int(roi_normalized[1] * self.img_size)
        x2 = int(roi_normalized[2] * self.img_size)
        y2 = int(roi_normalized[3] * self.img_size)
        return [x1, y1, x2, y2]
    

    def denormalize_rois(self, rois_normalized: List[List[float]]) -> List[List[int]]:
        if rois_normalized is None:
            return None
        rois_pixels = []
        for roi_norm in rois_normalized:
            x1 = int(roi_norm[0] * self.img_size)
            y1 = int(roi_norm[1] * self.img_size)
            x2 = int(roi_norm[2] * self.img_size)
            y2 = int(roi_norm[3] * self.img_size)
            rois_pixels.append([x1, y1, x2, y2])
        return rois_pixels


    # ---------------------- Detection on frames, this version allows visualization for debugging ----------------------
    @profiled
    def run(self, video_path: str = None, save_path: Optional[str] = None, show_video: bool = False):
        decoder = FrameDecoder(video_path, frame_skip=self.st_frame_skip, resize=self.st_resize)
        self.fps = decoder.fps or 16
        self.width, self.height = self._get_video_resolution(video_path=video_path)

        # Data preparation
        self.movement_info = {
            cls: {
                "detected": False,
                "movement_count": 0,
                "movements": [],
                "_active": False,
                "_start_frame": None,
                "_start_time": None
            }
            for cls in self.st_target_classes
        }

        # Default background frame selection (if None has been provided)
        if self.background_frame is None:
            print('No path for background frame, using first frame')
            for idx, _, frame in decoder.frames():
                if idx == self.pw_background_frame_index:
                    self.background_frame = frame.copy()
                    break

        self.background_frame = cv2.resize(self.background_frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        last_timestamp = 0.0
        self.pw_sequence_idx = 0
        self.pw_classes_in_sequence = {'detected_classes': {}, 'primary_class': None}

        # --- Main Loop ---
        for idx, timestamp, frame in decoder.frames():
            
            last_timestamp = timestamp
            (
                self.movement_info,
                self.pw_sequence_idx,
                self.pw_classes_in_sequence
            ) = self.process_frame(
                idx, timestamp, frame, 
                self.movement_info,
                self.background_frame, 
                show_video
            ) 
            if show_video:
                cv2.imshow("Detections", self.combined_display)
                key = cv2.waitKey(int(self.fps)) & 0xFF
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

        # Close ongoing immobility at the end of the sequence:
        self.close_ongoing_immobility(last_timestamp, idx)
        # Close ongoing tracking at video at the end of the sequence:
        self.close_ongoing_tracking()
        
        self.pw_sequences = [
            seq for seq in self.pw_sequences
            if seq['detected_classes'].get(seq['primary_class'], 0) >= self.pw_min_class_count
        ]
        
        return {
            "movement_info": self.movement_info,
            "pw_sequences": self.pw_sequences
        }

    @profiled
    def process_frame(
        self, idx, timestamp, frame, movement_info,
        background_frame, show_video: bool = False
    ):
        # Resizing using the same size that was used for training
        frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        if show_video:
            self.vis_frame1 = frame.copy()
            self.vis_frame2=frame.copy()

        # --- PersistentObjectWatcher immobility ---
        self.process_pw(background_frame=background_frame, frame=frame, timestamp=timestamp, idx=idx, show_video=show_video)

        # --- YOLO detections ---
        self.detections = self.detect(frame)

        # --- Persistent watcher classification ---
        self.process_pw_classification(show_video=show_video)

        # --- Simple Tracking ---          
        self.process_st(timestamp=timestamp, idx=idx, show_video=show_video)

        if show_video:
            if self.st_roi:
                for roi in self.st_roi:
                    
                    cv2.rectangle(self.vis_frame1, roi[:2], roi[2:], (100, 255, 100), 2)
            if self.pw_roi:
                cv2.rectangle(self.vis_frame2, self.pw_roi[:2], self.pw_roi[2:], (100, 255, 100), 2)
            # Resize + Display
            if self.display_scale != 1.0:
                self.vis_frame1 = cv2.resize(self.vis_frame1, (0, 0), fx=self.display_scale, fy=self.display_scale)
                self.vis_frame2 = cv2.resize(self.vis_frame2, (0, 0), fx=self.display_scale, fy=self.display_scale)
        
            self.combined_display = np.hstack((self.vis_frame1, self.vis_frame2))
            
        return movement_info, self.pw_sequence_idx, self.pw_classes_in_sequence
    
    @profiled
    def detect(self, frame):
        detections = self.detector.detect_frame(frame)
        return detections
    
    def process_pw(self, background_frame, frame,timestamp,idx, show_video):
        x1, y1, x2, y2 = self.pw_roi if self.pw_roi else (0, 0, background_frame.shape[1], background_frame.shape[0])
        bg_roi = background_frame[y1:y2, x1:x2]

        roi_frame = frame[y1:y2, x1:x2]
        diff = cv2.absdiff(roi_frame, bg_roi)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        _, fg_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        # Compute similarity with previous mask
        sim = self.compute_mask_similarity(fg_mask, self.pw_prev_mask) if self.pw_prev_mask is not None else 0.0
        self.pw_prev_mask = fg_mask.copy()
        self.pw_similarity_window.append(sim)

        if len(self.pw_similarity_window) >= 7:
            high_count = sum(1 for s in self.pw_similarity_window if s >= self.pw_high_similarity_threshold)
            low_count = sum(1 for s in self.pw_similarity_window if s < self.pw_low_similarity_threshold)
            
            # Start of an immobility sequence
            if not self.pw_immobile and high_count >= self.pw_min_high_frames:
                self.pw_immobile = True
                self.pw_immobile_start_frame = idx - 6
                self.pw_immobile_start_timestamp = timestamp - 6 / self.fps
            # End of an immobility sequence
            elif self.pw_immobile and low_count >= self.pw_min_low_frames:
                self.pw_immobile = False
                duration = timestamp - self.pw_immobile_start_timestamp
                if duration >= self.pw_min_immobility_duration:
                    self.pw_sequences.append({
                        'sequence_idx': self.pw_sequence_idx,
                        'start_frame': self.pw_immobile_start_frame,
                        'end_frame': idx,
                        'start_timestamp': self.pw_immobile_start_timestamp,
                        'end_timestamp': timestamp,
                        'duration': duration
                    })
                    self.pw_sequence_just_ended = True
                    
                self.pw_immobile_start_frame = None
                self.pw_immobile_start_timestamp = None
            if self.pw_immobile and show_video:
                color = (0, 0, 255) if self.pw_immobile else (0, 255, 0)
                cv2.rectangle(self.vis_frame2, (x1, y1), (x2, y2), color, 2)
                status = "Immobile" if self.pw_immobile else "Mobile"
                cv2.putText(self.vis_frame2, f"State: {status}", (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(self.vis_frame2, f"Sim: {sim:.2f}", (x1 + 10, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    def process_pw_classification(self, show_video=False):
        for cls in self.pw_target_classes:
            if self.pw_immobile or self.pw_sequence_just_ended:
                pw_detections = [
                    d for d in self.detections if d['class_name'] == cls and self._is_significantly_inside_roi(d['bbox'], self.pw_roi)
                ]
                if pw_detections:
                    pw_target = max(pw_detections, key=lambda d: d['conf'])
                    bbox=pw_target['bbox']
                    if pw_target['class_name'] in self.pw_classes_in_sequence['detected_classes']:
                        self.pw_classes_in_sequence['detected_classes'][pw_target['class_name']] += 1
                    else:
                        self.pw_classes_in_sequence['detected_classes'][pw_target['class_name']] = 1
                    if show_video:
                        cv2.rectangle(self.vis_frame2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.class_colors[cls], 2)
                        cv2.putText(self.vis_frame2, f"class {pw_target['class_name']}", (bbox[0] + 10, bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.class_colors[cls], 2)

                if self.pw_sequence_just_ended:  # End of sequence
                    if self.pw_classes_in_sequence['detected_classes']:
                        self.pw_classes_in_sequence['primary_class'] = max(
                            self.pw_classes_in_sequence['detected_classes'],
                            key=self.pw_classes_in_sequence['detected_classes'].get
                        )
                    else :
                        self.pw_classes_in_sequence['detected_classes']['no_classes_detected'] = self.pw_min_class_count
                        self.pw_classes_in_sequence['primary_class'] = 'no_classes_detected'
                    self.pw_sequences[self.pw_sequence_idx]['detected_classes'] = self.pw_classes_in_sequence['detected_classes']
                    self.pw_sequences[self.pw_sequence_idx]['primary_class'] = self.pw_classes_in_sequence['primary_class']
                    self.pw_classes_in_sequence = {'detected_classes': {}, 'primary_class': None}
                    self.pw_sequence_idx += 1
                    self.pw_sequence_just_ended = False
                    
    def process_st(self, timestamp, idx, show_video=False):
        
        for cls in self.st_target_classes:
            target_dets = [d for d in self.detections if d['class_name'] == cls and self._is_inside_rois(d['bbox'], self.st_roi)]
            bbox= None
            center= None
            smooth_center= None
            if target_dets:
                target = max(target_dets, key=lambda d: d['conf'])
                bbox = target['bbox']
                center = self._bbox_center(bbox)
                smooth_center = self._smooth_position(cls, center)
                self.st_valid_positions[cls]= [p for p in self.st_positions[cls] if p is not None]
            else:
                smooth_center = self._smooth_position(cls, None)
            self.st_tracking_data[cls].append({
                "frame_idx": idx,
                "timestamp": timestamp,
                "bbox": bbox,
                "center": smooth_center
            })
            moving = self._detect_movement_pattern(cls, smooth_center)
            cls_state = self.movement_info[cls]

            if moving and not cls_state['_active']:
                cls_state['_active'] = True
                cls_state['_start_frame'] = idx
                cls_state['_start_time'] = timestamp
            elif not moving and cls_state['_active']:
                cls_state['_active'] = False
                cls_state['movements'].append({
                    "start_frame": cls_state['_start_frame'],
                    "end_frame": idx,
                    "start_time": cls_state['_start_time'],
                    "end_time": timestamp
                })
                cls_state["movement_count"] += 1
                cls_state["detected"] = True
            if cls_state['_active'] and show_video:
                msg=self.alert_messages[cls]
                cv2.putText(
                    self.vis_frame1, msg, (30, 50 + 40 * list(self.st_target_classes).index(cls)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.class_colors[cls], 3, cv2.LINE_AA
                )
            if show_video and bbox is not None:

                    # Tracker visualization:

                cv2.rectangle(self.vis_frame1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.class_colors[cls], 2)
                cx, cy = map(int, smooth_center)
                cv2.circle(self.vis_frame1, (cx, cy), 5, self.class_colors[cls], -1)

                # Trajectory
                pts = self.st_positions[cls]
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue  # skip frames with unvalid points (ie no detections)
                    p1 = tuple(map(int, pts[i - 1]))
                    p2 = tuple(map(int, pts[i]))
                    cv2.line(self.vis_frame1, p1, p2, self.class_colors[cls], 2)

    # Closing ongoing immobility at the end of a sequence
    def close_ongoing_immobility(self, last_timestamp, idx):
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
                if self.pw_classes_in_sequence['detected_classes']:
                    self.pw_classes_in_sequence['primary_class'] = max(
                        self.pw_classes_in_sequence['detected_classes'],
                        key=self.pw_classes_in_sequence['detected_classes'].get
                    )
                else :
                    self.pw_classes_in_sequence['detected_classes']['no_classes_detected'] = self.pw_min_class_count
                    self.pw_classes_in_sequence['primary_class'] = 'no_classes_detected'
                self.pw_sequences[self.pw_sequence_idx]['detected_classes'] = self.pw_classes_in_sequence['detected_classes']
                self.pw_sequences[self.pw_sequence_idx]['primary_class'] = self.pw_classes_in_sequence['primary_class']
                self.pw_classes_in_sequence = {'detected_classes': {}, 'primary_class': None}
                self.pw_sequence_idx += 1
                self.pw_sequence_just_ended = False

            self.pw_immobile_start_frame = None
            self.pw_immobile_start_timestamp = None
            self.pw_immobile = False
        
    # Closing of ongoing tracking at the end of a sequence 
    def close_ongoing_tracking(self):
        for cls, cls_state in self.movement_info.items():
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

    def reset_sequences_variables(self):
        self.pw_sequence_idx = 0
        self.pw_sequence_just_ended = False
        self.pw_immobile = False
        self.pw_classes_in_sequence = {'detected_classes': {}, 'primary_class': None}
        self.pw_prev_mask = None
        self.pw_similarity_window = deque(maxlen=7)
        self.pw_immobile = False
        self.pw_immobile_start_frame = None
        self.pw_immobile_start_timestamp = None
        self.pw_sequences = []
        
        self.st_positions = {cls: deque(maxlen=self.st_smoothing_window) for cls in self.st_target_classes}
        self.st_valid_positions= {cls: [] for cls in self.st_target_classes}
        self.st_deviation_flags = {cls: deque(maxlen=self.st_movement_window) for cls in self.st_target_classes}
        self.st_tracking_data = {cls: [] for cls in self.st_target_classes}


    def run_on_folder(self, folder_path, show_video=False):

        folder_path = Path(folder_path)
        results_dict = {}

        for video_file in folder_path.glob("*.mp4"):
            print(f"Running on video: {video_file.name}...")
            result = self.run(str(video_file), show_video=show_video)
            
            self.reset_sequences_variables()
            results_dict[video_file.name] = result

        return results_dict
    
    # Sort the videos by detection
    def sort_videos_by_detection(self, results_dict, source_folder, output_folder):
   
        source_folder = Path(source_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)

        movement_classes = self.st_target_classes
        immobility_classes =self.pw_target_classes

        for video_name, result in results_dict.items():
            video_path = source_folder / video_name

            if not video_path.exists():
                print(f"Video not found: {video_name}")
                continue

            destinations = set()

            # ---- Detected Movements ----
            movement_info = result.get("movement_info", {})
            for obj in movement_classes:
                if movement_info.get(obj, {}).get("detected"):
                    destinations.add(f"mvt_{obj}")

            # ---- Detected Immobility ----
            pw_sequences = result.get("pw_sequences", [])
            detected_classes = set()
            for seq in pw_sequences:
                detected_classes.update(seq.get("detected_classes", {}).keys())

            for obj in immobility_classes:
                if obj in detected_classes:
                    destinations.add(f"immobility_{obj}")

            # ---- If no detection → potential_real_alert ----
            if not destinations:
                destinations.add("potential_real_alert")

            # ---- Copy video into each associated folder ----
            for dest in destinations:
                dest_path = output_folder / dest
                dest_path.mkdir(exist_ok=True, parents=True)
                shutil.copy2(video_path, dest_path / video_name)
                print(f"{video_name} → {dest_path}")

        print("Sorting complete.")
    

if __name__ == "__main__":
    detector=Detector("config.yaml")
    unified_watcher = UnifiedWatcher("config.yaml", detector=detector)
    results=unified_watcher.run_on_folder(folder_path="videos/PDV15", show_video=False)
    unified_watcher.sort_videos_by_detection(results_dict=results, source_folder='videos/PDV15', output_folder='videos/PDV15_sortedv2')
    plot_profile_stats(smoothing_window=5)