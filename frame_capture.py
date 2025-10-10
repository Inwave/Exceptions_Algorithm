import os
import cv2
from pathlib import Path
from typing import Generator, Tuple, Optional, List, Dict, Any


class FrameDecoder:
    """
    Iterates over frames in a video using OpenCV.
    """

    def __init__(self, video_path: str, frame_skip: int = 0, resize: Optional[Tuple[int, int]] = None):
        self.video_path = str(video_path)
        self.frame_skip = int(frame_skip)
        self.resize = resize

        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def frames(self) -> Generator[Tuple[int, float, Any], None, None]:
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.frame_skip and (idx % (self.frame_skip + 1) != 0):
                idx += 1
                continue

            if self.resize:
                frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)

            timestamp = idx / self.fps if self.fps > 0 else 0.0
            yield idx, timestamp, frame
            idx += 1

    def release(self):
        if self.cap:
            self.cap.release()
