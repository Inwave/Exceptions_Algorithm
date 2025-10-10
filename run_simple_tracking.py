from detection_analysis import Detector
from simple_tracking import SimpleTracking

# Run tracking from local source

detector = Detector("models/final_best_103.pt", conf_threshold=0.35)
tracker = SimpleTracking(
    video_path="videos/video_CarrefourSP_TBE1_20250916T195008_15.mp4",
    detector=detector,
    movement_threshold=10,
    roi=[[194, 172, 1000, 1000],[334, 164, 412, 258]],
    target_classes=['scanner','rag']

)


if __name__ == "__main__":
    result = tracker.run(verbose=True)
    print("Movement detected:", result["movement_detected"])