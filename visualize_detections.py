import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from ultralytics import YOLO

# Function drawing bounding boxes detected by YOLO model only on video.


DEFAULT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255)
]

def draw_detections(
    frame: np.ndarray,
    boxes: List[List[float]],
    clss: List[int],
    confs: List[float],
    names: List[str],
    colors: List[tuple] = DEFAULT_COLORS
) -> np.ndarray:
    """Draw bouding boxes and labels on frame."""
    for (x1, y1, x2, y2), c, p in zip(boxes, clss, confs):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        lbl = f"{names[int(c)]} {p:.2f}"
        color = colors[int(c % len(colors))]  # Évite les index hors limites
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame,
            (x1, max(0, y1 - th - 6)),
            (x1 + tw + 6, y1),
            color,
            -1
        )
        cv2.putText(
            frame, lbl, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1, cv2.LINE_AA
        )
    return frame

class RealTimeVisualizer:
   
    def __init__(
        self,
        input_path: str,
        model_path:str,
        output_dir: str = "annotated_videos",
        classes: Optional[List[str]] = None,
        colors: List[tuple] = DEFAULT_COLORS,
    ):
        """
        :param input_path: Path toward videos
        :param output_dir: output dir for annotated videos
        :param classes: Classes names
        :param colors: Colors for bounding boxes
        :param detect_frame: Function detecting frames
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.classes = classes or []
        self.colors = colors
        self.model_path=model_path
        self.model=YOLO(self.model_path)

    def annotate_video(
        self,
        video_path: str,
        output_path: Path,
    ) -> None:
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Impossible to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.detect_frame(frame) is not None:
                boxes, clss, confs = self.detect_frame(frame)
                if boxes:  
                    frame = draw_detections(
                        frame, boxes, clss, confs, self.classes, self.colors
                    )

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        print(f"[DONE] Annotated video saved: {output_path}")

    def detect_frame(self,frame):
        results = self.model(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().tolist()
        clss = results.boxes.cls.cpu().tolist()
        confs = results.boxes.conf.cpu().tolist()
        return boxes, clss, confs

    def process_all(self) -> None:
        
        if self.input_path.is_file():
            videos = [self.input_path]
        elif self.input_path.is_dir():
            videos = sorted(list(self.input_path.glob("*.mp4")))
        else:
            raise FileNotFoundError(f"Chemin introuvable: {self.input_path}")

        for vid_path in videos:
            name = vid_path.stem
            output_path = self.output_dir / f"{name}_test_annotated.mp4"
            self.annotate_video(vid_path, output_path)

    


if __name__ == "__main__":

    # Initialization
    visualizer = RealTimeVisualizer(
        input_path="videos/video_CarrefourSP_TBE1_20250916T195008_15.mp4",
        output_dir="annotated_videos",
        classes=["scanner", "hand", "price-sheets", "plastic-bags", "rag", "products"],
        model_path='models/best.pt'  # Model path
    )

    # Lanch
    visualizer.process_all()
