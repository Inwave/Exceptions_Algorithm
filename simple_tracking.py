from frame_capture import FrameDecoder
from detection_analysis import Detector

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
import datetime
import yaml


DEFAULT_COLORS = {
    'scanner':(255, 0, 0),'hand': (0, 255, 0),'rag' :(0, 0, 255),'plastic-bag':(255, 255, 0), 'products':(0, 255, 255), 'price-sheets':(255, 0, 255)}


class SimpleTracking:
    def __init__(
        self,
        config_path: str,
        detector: Any,    
    ):
        config = self.load_config(config_path)
        self.video_path = config.get("video_path", 'video/path/mp4')
        self.target_classes = config.get("target_classes", [])
        self.roi = config.get("roi", None)
        self.frame_skip = config.get("frame_skip", 0)
        self.resize = None
        self.movement_threshold = config.get("movement_threshold", 0.015)
        self.smoothing_window = config.get("smoothing_window", 25)
        self.alpha = config.get("alpha", 0.8)
        self.min_movement_count = config.get("min_movement_count", 5)
        self.movement_window = config.get("movement_window", 12)
        self.display_scale = config.get("display_scale", 1)
        self.detector = detector

        # ✅ State per class
        self.positions = {cls: deque(maxlen=self.smoothing_window) for cls in self.target_classes}
        self.deviation_flags = {cls: deque(maxlen=self.movement_window) for cls in self.target_classes}
        self.tracking_data = {cls: [] for cls in self.target_classes}
        self.movement_display_timer = {cls: 0 for cls in self.target_classes}

        # Custom alert colors per class
        self.class_colors = DEFAULT_COLORS

        # Custom alert messages
        self.alert_messages = {
            cls: f"MOVEMENT DETECTED: {cls.upper()}!" for cls in self.target_classes
        }
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("simpletracking", {})
    

    def _get_video_resolution(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (width, height)
    # -------------------------------------------------------------------------
    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _smooth_position(self, cls: str, new_pos):
        """Exponential smoothing per class."""
        if not self.positions[cls]:
            self.positions[cls].append(new_pos)
            return new_pos
        last_smoothed = self.positions[cls][-1]
        smoothed_x = self.alpha * new_pos[0] + (1 - self.alpha) * last_smoothed[0]
        smoothed_y = self.alpha * new_pos[1] + (1 - self.alpha) * last_smoothed[1]
        smoothed = (smoothed_x, smoothed_y)
        self.positions[cls].append(smoothed)
        return smoothed

    def _detect_movement_pattern(self, cls: str, current_center: Tuple[float, float]) -> bool:
        """Detect movement if several consecutive deviations exceed threshold."""
        if len(self.positions[cls]) < 2:
            return False

        mean_x = np.mean([p[0] for p in self.positions[cls]])
        mean_y = np.mean([p[1] for p in self.positions[cls]])
        mean_center = np.array([mean_x, mean_y])
        dist = np.linalg.norm(np.array(current_center) - mean_center)

        #Normalization:
        width, height = self._get_video_resolution()
        diagonal = np.sqrt(width ** 2 + height ** 2)

        normalized_dist = dist / diagonal

        self.deviation_flags[cls].append(normalized_dist > self.movement_threshold)
        count = sum(self.deviation_flags[cls])
        return count >= self.min_movement_count



    def _is_inside_roi(self, bbox: List[int]) -> bool:
        """Check if a bounding box intersects any ROI."""
        if self.roi is None:
            return True
        x1, y1, x2, y2 = bbox
        for roi in self.roi:
            rx1, ry1, rx2, ry2 = roi
            if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):
                return True
        return False

    # -------------------------------------------------------------------------------------
    def run(self, show_video: bool = True, save_path: Optional[str] = None, verbose: bool = True):

        decoder = FrameDecoder(self.video_path, frame_skip=self.frame_skip, resize=self.resize)
        movement_info = {
            cls: {
                "detected": False,
                "movement_count": 0,
                "movements": [],
                "_active": False,  # interne : indique si un mouvement est en cours
                "_start_frame": None,
                "_start_time": None
            }
            for cls in self.target_classes
        }

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(decoder.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.display_scale)
            height = int(decoder.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.display_scale)
            fps = decoder.fps or 30.0
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        for idx, timestamp, frame in decoder.frames():
            detections = self.detector.detect_frame(frame)
            vis_frame = frame.copy()

            for cls in self.target_classes:
                target_dets = [
                    d for d in detections
                    if d["class_name"] == cls and self._is_inside_roi(d["bbox"])
                ]
                if not target_dets:
                    continue

                # Most confident detection for that class
                target = max(target_dets, key=lambda d: d["conf"])
                bbox = target["bbox"]
                center = self._bbox_center(bbox)
                smooth_center = self._smooth_position(cls, center)
                self.tracking_data[cls].append({
                    "frame_idx": idx,
                    "timestamp": timestamp,
                    "bbox": bbox,
                    "center": smooth_center,
                    "conf": target["conf"],
                })

                # Detect movement
                is_moving = self._detect_movement_pattern(cls, smooth_center)

                cls_state = movement_info[cls]

                # --- Start of global movement ---
                if is_moving and not cls_state["_active"]:
                    cls_state["_active"] = True
                    cls_state["_start_frame"] = idx
                    cls_state["_start_time"] = timestamp
                    if verbose:
                        print(f"[Frame {idx}]  Starting movement '{cls}' à {timestamp:.2f}s")
                    

                elif not is_moving and cls_state["_active"]:
                    cls_state["_active"] = False
                    start_frame = cls_state["_start_frame"]
                    start_time = cls_state["_start_time"]
                    duration_s = timestamp - start_time
                    frame_count = idx - start_frame + 1

                    cls_state["movements"].append({
                        "start_frame": start_frame,
                        "end_frame": idx,
                        "start_timestamp": start_time,
                        "end_timestamp": timestamp,
                        "duration_s": duration_s,
                        "frame_count": frame_count
                    })
                    cls_state["movement_count"] += 1
                    cls_state["detected"] = True

                    if verbose:
                        print(f"[Frame {idx}] End of Movement '{cls}' at {timestamp:.2f}s "
                            f"(Duration: {duration_s:.2f}s, {frame_count} frames)")
                    
                if cls_state["_active"]:
                    msg = self.alert_messages[cls]
                    cv2.putText(
                        vis_frame, msg, (30, 50 + 40 * list(self.target_classes).index(cls)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA
                    )





                # --- Visualization ---
                color = self.class_colors[cls]
                cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cx, cy = map(int, smooth_center)
                cv2.circle(vis_frame, (cx, cy), 5, color, -1)

                # Trajectory
                pts = self.positions[cls]
                for i in range(1, len(pts)):
                    p1 = tuple(map(int, pts[i - 1]))
                    p2 = tuple(map(int, pts[i]))
                    cv2.line(vis_frame, p1, p2, color, 2)

                # Alert text
                if self.movement_display_timer[cls] > 0:
                    msg = self.alert_messages[cls]
                    cv2.putText(
                        vis_frame, msg, (30, 50 + 40 * list(self.target_classes).index(cls)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA
                    )
                    self.movement_display_timer[cls] -= 1

            # Draw ROI(s)
            if self.roi:
                for roi in self.roi:
                    cv2.rectangle(vis_frame, roi[:2], roi[2:], (100, 255, 100), 2)

            # Resize + Display
            if self.display_scale != 1.0:
                vis_frame = cv2.resize(vis_frame, (0, 0), fx=self.display_scale, fy=self.display_scale)

            if show_video:
                cv2.imshow("Tracking", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer:
                writer.write(vis_frame)

        for cls, cls_state in movement_info.items():
            if cls_state["_active"]:
                last_frame = self.tracking_data[cls][-1]["frame_idx"]
                last_time = self.tracking_data[cls][-1]["timestamp"]
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

            # Nettoyage des variables internes
            del cls_state["_active"]
            del cls_state["_start_frame"]
            del cls_state["_start_time"]

        if verbose:
            print("Movement detected:", movement_info)


        decoder.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        return {
            "movement_detected": movement_info,
            "tracking_data": self.tracking_data
        }





def draw_detections(frame, boxes, clss, confs, names, color=(0, 255, 0)):
    for (x1, y1, x2, y2), c, p in zip(boxes, clss, confs):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        lbl = f"{names[int(c)]} {p:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), DEFAULT_COLORS[names[c]], 2)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1),
                      DEFAULT_COLORS[names[c]], -1)
        cv2.putText(frame, lbl, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_boxes(frame, boxes, name='ROI'):
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        lbl = f"{name}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,180), 2)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1),
                    (255,0,180), -1)
        cv2.putText(frame, lbl, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)




class RTSPSimpleTracker:
    """YOLO-based simple tracker for RTSP streams with movement alert."""
    def __init__(self, model, target_classes: List[str],roi: Optional[List[Tuple[int, int, int, int]]] = None, conf_threshold=0.3,
                 movement_threshold=30.0, min_movement_count=3,
                 movement_window=8, alpha=0.6, smoothing_window=5, display_scale=1.0):
        self.model = model
        self.target_classes = target_classes
        self.conf_threshold = conf_threshold
        self.movement_threshold = movement_threshold
        self.min_movement_count = min_movement_count
        self.movement_window = movement_window
        self.smoothing_window = smoothing_window
        self.alpha = alpha
        self.roi=roi

        self.smoothed_center = None
        self.positions = {}
        self.deviation_flags = {}        
        self.movement_display_timer = {}
    
    def _get_video_resolution(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (width, height)
    
    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _smooth_position(self, new_pos, cls_name):
        if not self.positions[cls_name]:
            self.positions[cls_name].append(new_pos)
            return new_pos
        last = self.positions[cls_name][-1]
        smoothed = (
            self.alpha * new_pos[0] + (1 - self.alpha) * last[0],
            self.alpha * new_pos[1] + (1 - self.alpha) * last[1]
        )
        self.positions[cls_name].append(smoothed)
        return smoothed
    
    def _is_inside_roi(self, bbox: List[int]) -> bool:
        """Check if a bounding box intersects with the ROI."""
        if self.roi is None:
            return True
        x1, y1, x2, y2 = bbox
        for roi in self.roi:
            rx1, ry1, rx2, ry2 = roi
            if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):
                return True
        return False

    def _detect_movement_pattern(self, current_center, cls_name):
        if len(self.positions[cls_name]) < 2:
            return False

        mean_x = np.mean([p[0] for p in self.positions[cls_name]])
        mean_y = np.mean([p[1] for p in self.positions[cls_name]])
        mean_center = np.array([mean_x, mean_y])
        dist = np.linalg.norm(np.array(current_center) - mean_center)

        width, height = self._get_video_resolution()
        diagonal = np.sqrt(width ** 2 + height ** 2)

        normalized_dist = dist / diagonal

        self.deviation_flags[cls_name].append(normalized_dist > self.movement_threshold)
        count = sum(self.deviation_flags[cls_name])
        return count >= self.min_movement_count


    def process_frame(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        r = results[0]

        if self.roi is not None:
            draw_boxes(frame, self.roi)

        # No detections
        if not r.boxes or len(r.boxes) == 0:
            return frame, {}

        boxes = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        names = r.names if hasattr(r, "names") else self.model.names
        all_tracked_boxes = []
        movement_info = {
            cls: {
                "detected": False,
                "movement_count": 0,
                "movements": [],
                "_active": False,  # interne : indique si un mouvement est en cours
                "_start_frame": None,
                "_start_time": None
            }
            for cls in self.target_classes
        }
        # --- loop through each target class ---
        for target_class in self.target_classes:
            # ensure per-class buffers exist
            if target_class not in self.positions:
                self.positions[target_class] = deque(maxlen=self.smoothing_window)
                self.deviation_flags[target_class] = deque(maxlen=self.movement_window)
                self.movement_display_timer[target_class] = 0
            
            # get detections of this class inside ROI
            target_indices = [
                i for i, c in enumerate(clss)
                if names[c] == target_class and self._is_inside_roi(boxes[i])
            ]

            if not target_indices:
                continue

            # pick the most confident detection for this class
            i = max(target_indices, key=lambda j: confs[j])
            bbox = boxes[i]
            conf = confs[i]
            all_tracked_boxes.append((bbox, clss[i], conf))

            # compute smoothed trajectory for this class
            center = self._bbox_center(bbox)
            smooth_center = self._smooth_position(center, target_class)

            # draw trajectory (per class color)
            color = DEFAULT_COLORS.get(target_class, (0, 255, 255))
            pts = list(self.positions[target_class])
            for k in range(1, len(pts)):
                p1 = tuple(map(int, pts[k - 1]))
                p2 = tuple(map(int, pts[k]))
                cv2.line(frame, p1, p2, color, 2)

            # detect movement per class
            is_moving = self._detect_movement_pattern(smooth_center, target_class)
            cls_state=movement_info[target_class]
            if is_moving and not cls_state['_active'] :
                cls_state["_active"] = True
                cls_state["_start_time"] = datetime.now()

            elif not is_moving and cls_state["_active"]:
                cls_state["_active"] = False
                start_time = cls_state["_start_time"]
                end_time = datetime.now()
                duration_s = end_time - start_time

                cls_state["movements"].append({
                    "start_timestamp": start_time,
                    "end_timestamp": end_time,
                    "duration_s": duration_s,
                })
                cls_state["movement_count"] += 1
                cls_state["detected"] = True
            if cls_state["_active"]:
                msg = self.alert_messages[target_class]
                cv2.putText(
                    frame, msg, (30, 50 + 40 * list(self.target_classes).index(target_class)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA
                )

                
                
                

        # --- draw all detections ---
        if all_tracked_boxes:
            draw_detections(
                frame,
                [b for b, _, _ in all_tracked_boxes],
                [c for _, c, _ in all_tracked_boxes],
                [cf for _, _, cf in all_tracked_boxes],
                names
            )

        # --- display class-specific alerts ---
        y_offset = 40
        for cls_name, timer in self.movement_display_timer.items():
            if timer > 0:
                color = DEFAULT_COLORS[cls_name]
                alert_text = f" Movement detected: {cls_name.upper()}"
                cv2.putText(frame, alert_text, (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
                self.movement_display_timer[cls_name] -= 1
                y_offset += 40

        return frame, movement_info

