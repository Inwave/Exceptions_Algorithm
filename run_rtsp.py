import argparse
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from simple_tracking import RTSPSimpleTracker


# Run tracking from RTSP flux

def open_rtsp(rtsp_url, width=None, height=None):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    parser = argparse.ArgumentParser(description="RTSP YOLO Tracker with movement alert")
    parser.add_argument("--rtsp", required=True, help="RTSP stream URL")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--movement-thresh", type=float, default=10.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model = YOLO(args.model)
    tracker = RTSPSimpleTracker(model,
                          target_classes=['scanner','rag','plastic-bag'],
                          conf_threshold=args.conf,
                          movement_threshold=args.movement_thresh,
                          roi=[[194, 172, 310, 245],[334, 164, 412, 258]])

    cap = open_rtsp(args.rtsp)
    win_name = "RTSP YOLO Tracker"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    fps_hist = deque(maxlen=30)
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Lost RTSP connection. Retrying...")
            time.sleep(1)
            cap.release()
            cap = open_rtsp(args.rtsp)
            continue

        processed, moved = tracker.process_frame(frame)

        # display FPS
        t_now = time.time()
        fps = 1.0 / max(1e-6, t_now - t_prev)
        t_prev = t_now
        fps_hist.append(fps)
        fps_avg = sum(fps_hist) / len(fps_hist)
        cv2.putText(processed, f"FPS: {fps_avg:.1f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 40), 2, cv2.LINE_AA)

        cv2.imshow(win_name, processed)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()