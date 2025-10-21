from detection_analysis import Detector
from simple_tracking import SimpleTracking

# Run tracking from local source

config_path='config.yaml'


detector = Detector(config_path=config_path)
tracker = SimpleTracking(
    config_path=config_path,
    detector=detector,
    video_path="videos/video_test_CarrefourSP_TBE1_20250924T155916_2.mp4"
   )


if __name__ == "__main__":
    result = tracker.run(verbose=True)
    print("Movement detected:", result["movement_info"])