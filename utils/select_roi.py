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


def grab_first_frame(cap, retries=5):
    for i in range(retries):
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
        time.sleep(0.5)
    raise RuntimeError("Impossible to get frame from source")


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
    print(f"\n✅ ROI sélectionnée : (x1={x1}, y1={y1}, x2={x2}, y2={y2})\n")
    return [x1, y1, x2, y2]


def main():
    parser = argparse.ArgumentParser(description="ROI Selector on RTSP flux or local video")
    parser.add_argument("--video", help="path of local video")
    parser.add_argument("--rtsp", help="URL RTSP")
    args = parser.parse_args()

    if not args.video and not args.rtsp:
        parser.error("Must precise --video <path> ou --rtsp <url>")

    is_rtsp = args.rtsp is not None
    source = args.rtsp if is_rtsp else args.video

    print(" Opening video source")
    cap = open_source(source, is_rtsp=is_rtsp)
    frame = grab_first_frame(cap)
    cap.release()

    roi = select_roi_from_frame(frame)

    if roi:
        
        print(f"ROI = {roi}")
    else:
        print("No ROI")


if __name__ == "__main__":
    main()
