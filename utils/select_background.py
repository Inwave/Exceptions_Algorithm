import cv2
import os
from datetime import datetime
import argparse


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

def select_background(source, is_rtsp=False, save_dir="backgrounds"):
    """
    Ouvre une vidéo (locale ou RTSP) et permet de sélectionner manuellement
    une frame à utiliser comme background.
    
    Commandes :
    - Flèche gauche / 'a' : frame précédente
    - Flèche droite / 'd' : frame suivante
    - Entrée : sélectionner cette frame comme background
    - Q : quitter sans sélection

    La frame sélectionnée est enregistrée automatiquement dans `save_dir`,
    et le chemin du fichier est affiché pour le copier dans config.yaml.
    """

    print("Opening video for background selection...")
    cap = open_source(source, is_rtsp=is_rtsp)

    frames = []
    current_idx = 0

    # Charger toutes les frames (attention à la RAM pour les longues vidéos)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        raise RuntimeError("No frames found in video")

    print(f"Loaded {total_frames} frames from source.")

    # Navigation manuelle
    while True:
        display_frame = frames[current_idx].copy()
        cv2.putText(display_frame, f"Frame {current_idx + 1}/{total_frames} - Left/Right to navigate, Enter=select, Q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Background Selector", display_frame)
        key = cv2.waitKey(0) & 0xFF

        if key == 81 or key == ord('a'):  # Left arrow / 'a'
            current_idx = max(0, current_idx - 1)
        elif key == 83 or key == ord('d'):  # Right arrow / 'd'
            current_idx = min(total_frames - 1, current_idx + 1)
        elif key == 13:  # Enter key
            cv2.destroyWindow("Background Selector")
            print(f" Selected background frame #{current_idx + 1}")

            # Enregistrer l’image
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"background_{timestamp}.jpg")
            cv2.imwrite(save_path, frames[current_idx])

            print(f"\n Background frame saved to: {save_path}")
            print("\n  You can add this to your config.yaml:\n")
            print(f"background_frame_path: \"{save_path}\"\n")

            return frames[current_idx], save_path

        elif key == ord('q'):
            cv2.destroyWindow("Background Selector")
            print(" No background frame selected")
            return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Background frame selector")
    parser.add_argument("--video", help="path of local video")
    parser.add_argument("--rtsp", help="URL RTSP")
    args = parser.parse_args()

    is_rtsp = args.rtsp is not None
    source = args.rtsp if is_rtsp else args.video

    background_frame, background_path = select_background(source, is_rtsp=is_rtsp)

    if background_path:
        print(" Frame saved and ready to use.")
    else:
        print("No background frame selected.")