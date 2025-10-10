# Exceptions_Algorithm
Algortihm handling some of the wrong alerts of the easi-nop algorithm

Currently, the tracker operates as a **single-object tracker** per target class. It estimates motion based on the smoothed trajectory of bounding box centers. This version does not perform multi-object identity tracking, which is planned for future development.

---

## Objectives
- Filter object detections based on:
  - Target classes of interest.
  - Regions of Interest (ROIs) corresponding to specific operational zones.
- Track the motion of detected objects over time.
- Detect and confirm significant movement events.
- Provide visual feedback for debugging and validation purposes.

---

## Algorithm Description

1. **Frame Acquisition**  
   Frames are read either from a local video file or a live RTSP stream.

2. **Object Detection**  
   The YOLO model processes each frame to produce bounding boxes, confidence scores, and class labels.

3. **Filtering**  
   - Only detections matching the specified `target_classes` are considered.  
   - Detections are further filtered by intersection with the configured ROI areas.
   - Only the bbox with the highest confidence rate is kept for each target_class .

4. **Tracking and Smoothing**  
   - The center of each bounding box is extracted per frame.  
   - The trajectory is smoothed using **exponential averaging** to reduce noise and stabilize the motion path.

5. **Movement Detection**  
   - The algorithm computes the deviation of the current smoothed position from the historical mean.  
   - When several consecutive deviations exceed a predefined threshold, a movement event is confirmed.  
   - A visual alert is displayed to indicate confirmed motion.

6. **Visualization and Output**  
   - Bounding boxes, ROIs, and trajectories are drawn on the output frames.  
   - Alerts are displayed when motion is detected.  
   - The output can be shown in real time or saved to a video file.

---

## Current Implementation
The current version supports:
- Multiple target classes, analyzed independently.
- Multiple ROIs per scene.
- Movement analysis based on a single, most confident detection per class and frame.

Limitations:
- No multi-object tracking (i.e., no persistent object identity).
- Sensitive to detection noise when multiple similar objects are present.

Future work includes integration with established multi-object tracking frameworks such as **DeepSORT** or **ByteTrack** to handle object re-identification and improve tracking robustness.

---
