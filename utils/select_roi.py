import argparse
import cv2
import time


def open_source(source, is_rtsp=False):
    if is_rtsp:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    else:
        cap = cv2.VideoCapture(source)
    return cap


def select_frame_for_roi(cap):
    """
    Let user scroll frame by frame to pick the frame to select ROI.
    Left/Right arrows: previous/next frame
    Enter: select current frame for ROI
    Q: quit without selection
    """
    frames = []
    current_idx = 0

    # Read all frames into memory for easy navigation (can optimize for long videos if needed)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    total_frames = len(frames)

    if total_frames == 0:
        raise RuntimeError("No frames found in video")

    while True:
        display_frame = frames[current_idx].copy()
        cv2.putText(display_frame, f"Frame {current_idx + 1}/{total_frames} - Left/Right to navigate, Enter=select, Q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Frame Selector", display_frame)
        key = cv2.waitKey(0) & 0xFF

        if key == 81 or key == ord('a'):  # Left arrow / 'a'
            current_idx = max(0, current_idx - 1)
        elif key == 83 or key == ord('d'):  # Right arrow / 'd'
            current_idx = min(total_frames - 1, current_idx + 1)
        elif key == 13:  # Enter key
            cv2.destroyWindow("Frame Selector")
            return frames[current_idx]
        elif key == ord('q'):
            cv2.destroyWindow("Frame Selector")
            return None



def select_roi_from_frame(frame):
    clone = frame.copy()
    cv2.putText(clone, "  Select Roi and press enter",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    roi = cv2.selectROI("ROI Selector", clone, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("ROI Selector")

    if all(v == 0 for v in roi):
        print("No ROI Selected")
        return None

    x, y, w, h = roi
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    print(f"\nROI selected : (x1={x1}, y1={y1}, x2={x2}, y2={y2})\n")
    return [x1, y1, x2, y2]


def main():
    parser = argparse.ArgumentParser(description="ROI Selector on RTSP stream or local video")
    parser.add_argument("--video", help="path of local video")
    parser.add_argument("--rtsp", help="URL RTSP")
    args = parser.parse_args()

    if not args.video and not args.rtsp:
        parser.error("Must specify --video <path> or --rtsp <url>")

    is_rtsp = args.rtsp is not None
    source = args.rtsp if is_rtsp else args.video

    print("Opening video source...")
    cap = open_source(source, is_rtsp=is_rtsp)

    frame = select_frame_for_roi(cap)
    cap.release()

    if frame is not None:
        roi = select_roi_from_frame(frame)
        if roi:
            print(f"ROI = {roi}")
        else:
            print("No ROI selected")
    else:
        print("No frame selected for ROI")

if __name__ == "__main__":
    main()
