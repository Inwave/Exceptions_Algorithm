import time

start_time = time.time()

import cv2
import yaml
import numpy as np
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from frame_capture import FrameDecoder
from detection_analysis import Detector
from profiler import profiled

end_time = time.time()
print('import time', end_time - start_time)

class PersistentObjectWatcher:
    def __init__(
        self,
        config_path: str,
        detector: Any,
        video_path: str
    ):
        config = self.load_config(config_path)
        self.video_path = video_path
        self.detector = detector

        # ROI representing the scale zone
        self.roi = config.get("roi", None)

        # Time threshold for "too long" in seconds
        self.time_threshold = config.get("time_threshold", 5.0)

        # Movement filtering
        self.smoothing_window = config.get("smoothing_window", 10)
        self.alpha = config.get("alpha", 0.8)
        self.motion_threshold = config.get("motion_threshold", 0.01)
        self.disappearance_window= config.get("disappearance_window", 10)
        # Class filters
        self.target_classes = config.get("target_classes", [])


        # Visualization
        self.display_scale = config.get("display_scale", 1.0)
        self.alert_color = (0, 0, 255)  # red

        # Internal states
        self.tracked_objects = {}  # {id: {"class": str, "last_seen": t, "in_roi": bool, "enter_time": t}}
        self.next_id = 0
        self.active_alerts = []

        self.width, self.height = self._get_video_resolution()

    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("persistentwatcher", {})
    @profiled
    def _get_video_resolution(self) -> Tuple[int, int]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return (0, 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (width, height)

    def _bbox_center(self, bbox: List[int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2
    
    @staticmethod
    def ioa(bbox: List[int], roi: List[int]) -> float:
        #intersection_over_smallest_area
        ix1 = max(bbox[0], roi[0])
        iy1 = max(bbox[1], roi[1])
        ix2 = min(bbox[2], roi[2])
        iy2 = min(bbox[3], roi[3])
        intersection_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        roi_area = (roi[2] - roi[0]) * (roi[3] - roi[1])
        smallest_area = min(bbox_area, roi_area)
        if smallest_area == 0:
            return 0.0

        return intersection_area / smallest_area
    @profiled
    def _is_inside_roi(self, bbox: List[int]) -> bool:
        if self.roi is None:
            return False
        x1, y1, x2, y2 = bbox
        rx1, ry1, rx2, ry2 = self.roi
        return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)
    
    def is_sufficiently_in_roi(self, bbox: List[int]) -> bool:
        if self.ioa(bbox, self.roi) < 0.5:
            return False
        return True

    @staticmethod
    def compute_mask_similarity(fg_mask, prev_mask,min_area_ratio=0.1):
        
        total_area = fg_mask.shape[0] * fg_mask.shape[1]
        min_pixels = int(total_area * min_area_ratio)
        
        intersection = cv2.bitwise_and(fg_mask, prev_mask)
        union = cv2.bitwise_or(fg_mask, prev_mask)
        
        intersection_area = cv2.countNonZero(intersection)
        union_area = cv2.countNonZero(union)
        
        if union_area < min_pixels:
            return 0.0
        
        iou = intersection_area / union_area
        
        return iou
    @profiled
    def detect_background_based_immobility(
        self,
        show_video=True,
        show_masks=True,
        save_path=None,
        display_scale=0.5,
        high_similarity_threshold=0.9,  # high threshold to detect immobility start
        low_similarity_threshold=0.7,   # low threshold to detect immobility end
        min_high_frames=5,              # k frames above high threshold
        min_low_frames=3,               # k frames below low threshold
        window_size=7,                 # n last frames to consider
        min_immobile_duration=1.5,      # minimum duration in seconds to record
        background_frame_index=0,
        min_area_ratio=0.1            # minimum ratio to filter noise
    ):
        """
        Detects immobility in a ROI based on persistent background changes.
        Uses mask similarity across frames with hysteresis thresholds.
        
        Args:
            high_similarity_threshold: Threshold to start immobility
            low_similarity_threshold: Threshold to end immobility
            min_high_frames: Number of frames above high threshold over window_size
            min_low_frames: Number of frames below low threshold over window_size
            window_size: Size of the sliding window
            min_immobile_duration: Minimum duration to validate a sequence
            min_area_ratio: Minimum pixel ratio to filter noise
        """

        from frame_capture import FrameDecoder
        from collections import deque
        
        decoder = FrameDecoder(self.video_path)
        fps = decoder.fps or 8
        background_frame = None
        for idx, _, frame in decoder.frames():
            if idx == background_frame_index:
                background_frame = frame.copy()
                break
        
        # Output video writer
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(decoder.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * display_scale)
            height = int(decoder.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * display_scale)
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        # ROI setup
        bg_resized = cv2.resize(background_frame, (0, 0), fx=display_scale, fy=display_scale)
        x1, y1, x2, y2 = [int(v * display_scale) for v in self.roi]
        roi_width, roi_height = x2 - x1, y2 - y1
        bg_roi = bg_resized[y1:y2, x1:x2]

        # Detection state
        prev_mask = None
        similarity_window = deque(maxlen=window_size)
        immobile = False
        immobile_start_frame = None
        immobile_start_timestamp = None
        immobility_sequences = []
        last_timestamp = 0.0

        for idx, timestamp, frame in decoder.frames():
            last_timestamp = timestamp
            frame_disp = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
            roi_frame = frame_disp[y1:y2, x1:x2]

            # === Background subtraction ===
            diff = cv2.absdiff(roi_frame, bg_roi)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
            _, fg_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

            # === Compute similarity between consecutive masks ===
            if prev_mask is not None:
                sim = self.compute_mask_similarity(fg_mask, prev_mask, min_area_ratio=min_area_ratio)
            else:
                sim = 0.0
            prev_mask = fg_mask.copy()
            similarity_window.append(sim)

            # === Immobility detection logic ===
            if len(similarity_window) >= window_size:
                # Count frames above/below thresholds
                high_count = sum(1 for s in similarity_window if s >= high_similarity_threshold)
                low_count = sum(1 for s in similarity_window if s < low_similarity_threshold)

                # Detect immobility start
                if not immobile and high_count >= min_high_frames:
                    immobile = True
                    # Find the first frame that exceeds the threshold in the window
                    frames_back = 0
                    for i in range(len(similarity_window) - 1, -1, -1):
                        if similarity_window[i] >= high_similarity_threshold:
                            frames_back = len(similarity_window) - 1 - i
                    
                    immobile_start_frame = idx - frames_back
                    immobile_start_timestamp = timestamp - (frames_back / fps)
                    print(f"[Frame {idx}] Immobility started (from frame {immobile_start_frame})")

                # Detect immobility end
                elif immobile and low_count >= min_low_frames:
                    immobile = False
                    duration = timestamp - immobile_start_timestamp
                    
                    print(f"[Frame {idx}] Immobility ended (duration: {duration:.2f}s)")
                    
                    # Record only if duration exceeds threshold
                    if duration >= min_immobile_duration:
                        immobility_sequences.append({
                            'start_frame': immobile_start_frame,
                            'end_frame': idx,
                            'start_timestamp': immobile_start_timestamp,
                            'end_timestamp': timestamp,
                            'duration': duration
                        })
                        print(f"  → Sequence recorded: frames {immobile_start_frame}-{idx}")
                    else:
                        print(f"  → Sequence ignored (duration < {min_immobile_duration}s)")
                    
                    immobile_start_frame = None
                    immobile_start_timestamp = None

            # === Visualization ===
            if show_video or writer:
                color = (0, 0, 255) if immobile else (0, 255, 0)
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), color, 2)
                status = "Immobile" if immobile else "Mobile"
                cv2.putText(frame_disp, f"State: {status}", (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame_disp, f"Sim: {sim:.2f}", (x1 + 10, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            if show_masks:
                # Convert mask to BGR for side-by-side display
                mask_full = np.zeros_like(frame_disp)
                mask_full[y1:y2, x1:x2] = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                combined_display = np.hstack((frame_disp, mask_full))
            else:
                combined_display = frame_disp
                
            if writer:
                if show_masks:
                    writer.write(combined_display)
                else:
                    writer.write(frame_disp)
                    
            if show_video:
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

        # Close ongoing immobility at video end
        if immobile and immobile_start_frame is not None:
            duration = last_timestamp - immobile_start_timestamp
            if duration >= min_immobile_duration:
                immobility_sequences.append({
                    'start_frame': immobile_start_frame,
                    'end_frame': idx,
                    'start_timestamp': immobile_start_timestamp,
                    'end_timestamp': last_timestamp,
                    'duration': duration
                })
                print(f"[End] Final sequence recorded: frames {immobile_start_frame}-{idx}")

        if show_video:
            cv2.destroyAllWindows()
        if writer:
            writer.release()

        return {
            "has_immobility": len(immobility_sequences) > 0,
            "immobility_sequences": immobility_sequences
        }

    @profiled
    def detect_immobility_with_classification(
        self,
        show_video=False,
        save_path=None,
        display_scale=1.0,
        alert_color=(0, 0, 255),
        
        min_class_detections=3,  # minimum k detections of same class to confirm
        confidence_threshold=0.5  # YOLO confidence threshold
    ):
        """
        Detects immobility and classifies objects using YOLO.
        
        Args:
            min_class_detections: Minimum number of detections of the same class
                                 in a sequence to confirm object presence
            confidence_threshold: Minimum confidence for YOLO detections
            ... (other args from detect_background_based_immobility)
        
        Returns:
            dict with:
                - has_immobility: bool
                - immobility_sequences: list of dicts with:
                    - start_frame, end_frame, start_timestamp, end_timestamp, duration
                    - detected_classes: dict mapping class_name -> detection_count
                    - primary_class: most detected class in the sequence
        """
        
        from frame_capture import FrameDecoder
        from collections import defaultdict
        
        # First pass: detect immobility sequences
        immobility_result = self.detect_background_based_immobility(
            show_video=False,  # Don't show video during first pass
            show_masks=False,
            save_path=None,
            
        )
        
        if not immobility_result['has_immobility']:
            print("No immobility detected, skipping YOLO classification.")
            return immobility_result
        
        print(f"\nFound {len(immobility_result['immobility_sequences'])} immobility sequences.")
        print("Running YOLO classification on sequences...\n")
        
        # Second pass: classify objects in immobility sequences
        decoder = FrameDecoder(self.video_path)
        fps = decoder.fps or 8
        
        
        
        # Output video writer
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(decoder.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * display_scale)
            height = int(decoder.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * display_scale)
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        # ROI setup
        x1, y1, x2, y2 = [int(v * display_scale) for v in self.roi]
        
        # Create frame ranges for each sequence
        sequence_frames = set()
        sequence_map = {}  # maps frame_idx -> sequence_idx
        for seq_idx, seq in enumerate(immobility_result['immobility_sequences']):
            for frame_idx in range(seq['start_frame'], seq['end_frame'] + 1):
                sequence_frames.add(frame_idx)
                sequence_map[frame_idx] = seq_idx
            # Initialize class counters for each sequence
            seq['class_detections'] = defaultdict(int)
        
        # Process frames
        for idx, timestamp, frame in decoder.frames():
            frame_disp = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
            
            # Check if this frame is in an immobility sequence
            if idx in sequence_frames:
                seq_idx = sequence_map[idx]
                sequence = immobility_result['immobility_sequences'][seq_idx]
                
                # Run YOLO detection
                detections = self.detector.detect_frame(frame)
                
                # Filter detections in ROI and by confidence
                roi_detections = []
                for det in detections:
                    # Check if detection center is in ROI (adjust coordinates for display_scale)
                    x_center = (det['bbox'][0] + det['bbox'][2]) / 2 * display_scale
                    y_center = (det['bbox'][1] + det['bbox'][3]) / 2 * display_scale
                    
                    if (x1 <= x_center <= x2 and 
                        y1 <= y_center <= y2 and 
                        det['conf'] >= confidence_threshold):
                        roi_detections.append(det)
                        sequence['class_detections'][det['class_name']] += 1
                
                # Visualization
                if show_video or writer:
                    # Draw ROI
                    cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Draw detections
                    for det in roi_detections:
                        bbox = [int(coord * display_scale) for coord in det['bbox']]
                        x1_det, y1_det, x2_det, y2_det = bbox
                        
                        cv2.rectangle(frame_disp, (x1_det, y1_det), (x2_det, y2_det), 
                                    (0, 255, 0), 2)
                        
                        label = f"{det['class_name']}: {det['conf']:.2f}"
                        cv2.putText(frame_disp, label, (x1_det, y1_det - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Show sequence info
                    info_text = f"Sequence {seq_idx + 1} | Frame {idx}"
                    cv2.putText(frame_disp, info_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show current detections count
                    y_offset = 60
                    for class_name, count in sorted(sequence['class_detections'].items()):
                        text = f"{class_name}: {count}"
                        cv2.putText(frame_disp, text, (10, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        y_offset += 25
            else:
                # Not in immobility sequence
                if show_video or writer:
                    cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_disp, "No immobility", (x1 + 10, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if writer:
                writer.write(frame_disp)
            
            if show_video:
                cv2.imshow("YOLO Immobility Detection", frame_disp)
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
        
        if show_video:
            cv2.destroyAllWindows()
        if writer:
            writer.release()
        
        # Post-process: filter classes by minimum detections and find primary class
        for seq in immobility_result['immobility_sequences']:
            # Filter classes with enough detections
            confirmed_classes = {
                class_name: count 
                for class_name, count in seq['class_detections'].items()
                if count >= min_class_detections
            }
            
            seq['detected_classes'] = confirmed_classes
            
            # Find primary class (most detected)
            if confirmed_classes:
                seq['primary_class'] = max(confirmed_classes.items(), key=lambda x: x[1])[0]
            else:
                seq['primary_class'] = None
            
            # Clean up the raw counter
            del seq['class_detections']
            
            # Print summary
            print(f"Sequence {seq['start_frame']}-{seq['end_frame']} ({seq['duration']:.2f}s):")
            if confirmed_classes:
                print(f"  Primary class: {seq['primary_class']}")
                print(f"  All detected classes: {confirmed_classes}")
            else:
                print(f"  No objects detected with >= {min_class_detections} confirmations")
        
        return immobility_result
    
    

if __name__ == "__main__":
    config_path='config.yaml'
    detector=Detector(config_path)
    scale_detection=PersistentObjectWatcher(config_path, detector, 'videos/video_CarrefourSP_TBE1_20250917T164539_16.mp4')
    result=scale_detection.detect_immobility_with_classification(show_video=False)
    print(result)