# Exceptions_Algorithm
Algortihm handling some of the wrong alerts of the easi-nop algorithm

Currently, the tracker operates as a **single-object tracker** per target class. It estimates motion based on the smoothed trajectory of bounding box centers. This version does not perform multi-object identity tracking, which could be planned for future development. A persistent object watcher has been implemented.

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

Two algorithms are run simultaneously on the frames, a movement tracking algorithm, to detect if there's movement of non-products objects detected in selected ROIs, which might have raised a false alert. The second algorithm detects if there's an object immobile in a selected ROI (typically the scaling area), an event that might also raises false alerts.

1. **Frame Acquisition**  
   Frames are read either from a local video file or a live RTSP stream.

2. **Object Detection**  
   The YOLO model processes each frame to produce bounding boxes, confidence scores, and class labels.

### Movement tracking

3. **Filtering**  
   - Only detections matching the specified `st_target_classes` are considered.  
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

### Immobility detection

3. **Background Substraction**  
   - Compute absolute binary mask difference between the current roi and a reference background 
   - Compute the similarity between two following masks to detect immobility sequences
   - Keep the infos of an immobility sequence, with the starting and ending frames

4. **YOLO Detection**
   - Apply YOLO Detection if there is an immobility sequence to see what kind of object starteed the immobility sequence
   - Keep the primary class of the sequence if the number of detection is above min_class_detection

6. **Visualization and Output**  
   - Bounding boxes, ROIs, and trajectories are drawn on the output frames if there's an immobility sequence.  
   - The output can be shown in real time or saved to a video file (not yet implemented)

---

## Current Implementation
The current version supports:
- Multiple target classes, analyzed independently.
- Multiple ROIs per scene.
- Movement analysis based on a single, most confident detection per class and frame.
- Immobility Detection based on a combined background substraction and YOLO model detection

Limitations:
- No multi-object tracking (i.e., no real persistent object identity: the actual persistent object identity relies the fact that the object with the most confident detection is the same, but it might not be the case).
- Sensitive to detection noise when multiple similar objects are present.

Future work includes integration with established multi-object tracking frameworks such as **DeepSORT** or **ByteTrack** to handle object re-identification and improve tracking robustness.

---
