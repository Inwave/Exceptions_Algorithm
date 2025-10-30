# Exceptions_Algorithm
This algorithm handles some of the false positives alerts from the Easinop algorithm, which aims to reduce the number of misclassifications in shoplifting situations at store checkouts.

Currently, the tracker operates as a **single-object tracker** per target class. It estimates motion based on the smoothed trajectory of bounding box centers. The tracker focuses on the alerts that can be raised because of scanner, plastic_bags or rag movement in the 'scanning area' or the 'drop-off area'. This version does not perform multi-object identity tracking, which could be planned for future development. A **persistent object watcher** is also implemented, focusing on the alerts that can be raised because of scaling operation in the 'scaling area' by the POS operator.

---

## Objectives

- Filter object detections based on:
  - Target classes of interest.
  - Regions of Interest (ROIs) corresponding to specific operational zones.
- Track the motion of detected objects over time.
- Detect and confirm significant movement events.
- Detect scaling operations of products that can raise a false alert, using mask similarities and YOLO model.
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
   - The reference background is currently a frame on the format .jpg where there is nothing in the scaling roi. This must be adapted for each Point of Sales, and this is one of the main issue for the integration part.
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

### The current version supports:
- Multiple target classes, analyzed independently.
- Multiple ROIs per scene.
- Movement analysis based on a single, most confident detection per class and frame.
- Immobility Detection based on a combined background substraction and YOLO model detection

### Limitations and Remarks:

- The performance of the algorithm is mainly based on the **YOLO model performance**, a certain attention should be given to the training part. An idea for improvement would be to train a model for each store, improving performance but then the integration would be more time-consuming than a general model train once.
- Sensitive to detection noise when **multiple similar objects** are present, or when there is wrong detections with high confidence.
- A lot of parameters are specifics to each POS (Point of Sales (can be found as PDV: Ponte De Venda)) and must then be configured for each one (background frames, different ROIs)
- Config parameters performance can also differ between stores.
- When running on CPU, the more time consuming function is the **inference of the model detection** (93% of the process_frame execution time). On CPU the inference time on a frame is on average around 130 ms, which correspond to around processing around 6 fps. Running on each frames (frame_skip=0), the running time is approximatively 150% time slower than the actual videos of interest. This might be an issue in the future insertion in the easinop algorithm if GPU use can be provided. Setting frame_skip=1 allows to be faster than the video fps, but might decrease algorithm robustness.
- When running on a  NVIDIA GeForce RTX 4060 GPU, the inference time is around 15x faster, which allows the algorithm to be run without frame skip in a very reasonnable time to be insered in easinop.


Future work could include integration with established multi-object tracking frameworks such as **DeepSORT** or **ByteTrack** to handle object re-identification and improve tracking robustness.

---

## How to use

1. **Set up the virtual environment**  
   To install the necessary dependencies and activate the virtual environment, run the following commands in a terminal:
   
       pipenv install --dev
       pipenv shell

2. **Prepare computer vision models**  
   - Place the desired models inside the **`models/`** directory.  
   - Register each model path in the **`config.yaml`** file.

3. **Add input videos**  
   - Place all videos to be processed in the **`videos/`** directory.

4. **Configure point-of-view (POS) parameters**  
   The **`utils/`** folder provides tools to configure POS-specific parameters for the videos:
   - **`select_roi.py`** — define the Regions of Interest (ROIs) for scaling, scan, and drop-off areas.  
   - **`select_background.py`** — select and save the background image for the scaling area (saved as a `.jpg`).  
     Ensure you select a frame where the scaling area is clear (no objects or motion).

5. **Run the main algorithm**  
   Execute the following command to run the processing pipeline:
   
       python3 unified_watcher.py

   - To enable on-screen visualization, set `show_video=True` in the python file.

6. **Adjust algorithm parameters**  
   Edit **`config.yaml`** to tune algorithm behavior, including:
   - Model paths and types  
   - Threshold values  
   - Window sizes and other algorithm-specific parameters

7. **Batch processing and video sorting**  
   - Use the `run_on_folder` function to run the algorithm sequentially over multiple videos.  
   - After processing, videos can be sorted into different folders based on detected alerts using the `sort_videos_by_detection` function.

8. **Profiling and performance analysis**  
   A profiling utility is provided to identify the most time-consuming parts of the code and support performance optimization.
