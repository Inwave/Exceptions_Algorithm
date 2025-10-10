from frame_capture import FrameDecoder
from detection_analysis import Detector

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

class SimpleTracking:
    def __init__(
        self,
        video_path: str,
        detector: Any,
        target_classes: List[str],
        roi: Optional[List[Tuple[int, int, int, int]]] = None,  # list of [x1,y1,x2,y2]
        frame_skip: int = 0,
        resize: Optional[Tuple[int, int]] = None,
        movement_threshold: float = 30.0,
        smoothing_window: int = 5,
        alpha: float = 0.8,
        min_movement_count: int = 5,
        movement_window: int = 15,
        display_scale: float = 1.0,
    ):
        self.video_path = video_path
        self.detector = detector
        self.target_classes = target_classes
        self.roi = roi
        self.frame_skip = frame_skip
        self.resize = resize
        self.movement_threshold = movement_threshold
        self.smoothing_window = smoothing_window
        self.alpha = alpha
        self.min_movement_count = min_movement_count
        self.movement_window = movement_window
        self.display_scale = display_scale

        # ✅ State per class
        self.positions = {cls: deque(maxlen=smoothing_window) for cls in target_classes}
        self.deviation_flags = {cls: deque(maxlen=movement_window) for cls in target_classes}
        self.tracking_data = {cls: [] for cls in target_classes}
        self.movement_display_timer = {cls: 0 for cls in target_classes}

        # Custom alert colors per class
        self.class_colors = {
            cls: tuple(np.random.randint(50, 255, 3).tolist()) for cls in target_classes
        }

        # Custom alert messages
        self.alert_messages = {
            cls: f"MOVEMENT DETECTED: {cls.upper()}!" for cls in target_classes
        }

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

        self.deviation_flags[cls].append(dist > self.movement_threshold)
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

    # -------------------------------------------------------------------------
    def run(self, show_video: bool = True, save_path: Optional[str] = None, verbose: bool = True):

        decoder = FrameDecoder(self.video_path, frame_skip=self.frame_skip, resize=self.resize)
        movement_detected = {cls: False for cls in self.target_classes}

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
                if self._detect_movement_pattern(cls, smooth_center):
                    movement_detected[cls] = True
                    self.movement_display_timer[cls] = 15
                    if verbose:
                        print(f"[Frame {idx}] Movement confirmed for '{cls}' at {timestamp:.2f}s")

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

        decoder.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        return {
            "movement_detected": movement_detected,
            "tracking_data": self.tracking_data
        }

DEFAULT_COLORS = {
    'scanner':(255, 0, 0),'hand': (0, 255, 0),'rag' :(0, 0, 255),'plastic-bag':(255, 255, 0), 'products':(0, 255, 255), 'price-sheets':(255, 0, 255)}


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

        self.deviation_flags[cls_name].append(dist > self.movement_threshold)
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
        class_movements = {}

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
            moved = self._detect_movement_pattern(smooth_center, target_class)
            class_movements[target_class] = moved

            if moved:
                self.movement_display_timer[target_class] = 15

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
                alert_text = f"⚠️ Movement detected: {cls_name.upper()}"
                cv2.putText(frame, alert_text, (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
                self.movement_display_timer[cls_name] -= 1
                y_offset += 40

        return frame, class_movements

