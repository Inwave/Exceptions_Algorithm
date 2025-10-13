from ultralytics import YOLO
from typing import Generator, Tuple, Optional, List, Dict, Any
import cv2
import numpy as np
from time import time
from contextlib import suppress
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import yaml

from frame_capture import FrameDecoder


class Detector:
    def __init__(self,config_path):
        config=self.load_config(config_path)
        self.model_path = config.get("model_path", "")
        self.model = YOLO(self.model_path)
        self.conf_threshold = config.get("conf_threshold", 0.3)
        self.device = config.get("device", "cpu")
        self.classes = self.model.names
        
    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get('detector', {})

    def detect_frame(self, frame) -> List[Dict[str, Any]]:
        results = self.model(frame, verbose=False)[0]
        detections = []
        if not results or len(results) == 0:
            return detections

        preds = results.boxes
        if preds is None or preds.shape[0] == 0:
            return detections

        for box in preds:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.classes[cls_id] if (0 <= cls_id < len(self.classes)) else str(cls_id)
            detections.append(
                {"bbox": [x1, y1, x2, y2],
                 "conf": conf,
                 "class_id": cls_id,
                 "class_name": cls_name}
            )

        return detections
    

    def nms_max_overlap(
        self,
        detections: list[dict],
        iou_threshold: float = 0.75,
        class_filter: list[int] | None = None,
        roi: list[int] | None = None
    ) -> list[dict]:
        """
        Apply a class-aware NMS with optionnal class and region filtering

        Args:
            detections (list[dict]): (detect_frame)
            iou_threshold (float): treshold to consider overlapping
            class_filter (list[int], optional): list of class_id on which to apply NMS
            roi (list[int], optional): area [x1,y1,x2,y2] where to apply NMS (outside won't be filtered)
        Returns:
            list[dict]: new detections
        """

        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            inter_w = max(0, xB - xA)
            inter_h = max(0, yB - yA)
            inter_area = inter_w * inter_h
            if inter_area == 0:
                return 0.0
            areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            return inter_area / (areaA + areaB - inter_area)

        # Separation according to ROI
        inside, outside = [], []
        for det in detections:
            if roi:
                x1, y1, x2, y2 = det["bbox"]
                if (x1 >= roi[0] or y1 >= roi[1] or x2 <= roi[2] or y2 <= roi[3]):
                    inside.append(det)
                else:
                    outside.append(det)
            else:
                inside.append(det)

        # Application du NMS class-aware
        filtered = []
        classes_to_process = (
            class_filter if class_filter is not None else list(set(d["class_id"] for d in inside))
        )

        for cls in classes_to_process:
            cls_dets = [d for d in inside if d["class_id"] == cls]
            if not cls_dets:
                continue

            boxes = np.array([d["bbox"] for d in cls_dets])
            confs = np.array([d["conf"] for d in cls_dets])
            idxs = np.argsort(confs)[::-1]  # tri décroissant par confiance

            keep = []
            while len(idxs) > 0:
                i = idxs[0]
                keep.append(i)
                remaining = []
                for j in idxs[1:]:
                    if iou(boxes[i], boxes[j]) < iou_threshold:
                        remaining.append(j)
                idxs = np.array(remaining)
            filtered.extend([cls_dets[k] for k in keep])

        # Fusionner résultats (inside filtré + outside inchangé)
        final_detections = filtered + outside
        return final_detections

    