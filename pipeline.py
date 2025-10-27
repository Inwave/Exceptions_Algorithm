import shutil
import os
from typing import Dict, Any
from unified_watcher import UnifiedWatcher
from detection_analysis import Detector
class Pipeline:
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.output_dir = {
            "moving_scanners": os.path.join(video_dir, "moving_scanners"),
            "moving_plastic_bag": os.path.join(video_dir, "moving_plastic_bag"),
            "moving_rag": os.path.join(video_dir, "moving_rag"),
            "immobile_product": os.path.join(video_dir, "immobile_product"),
            "immobile_plastic_bag": os.path.join(video_dir, "immobile_plastic_bag"),
            "immobile_rag": os.path.join(video_dir, "immobile_rag"),
        }

        # Création des sous-dossiers si besoin
        for folder in self.output_dir.values():
            os.makedirs(folder, exist_ok=True)

    def run(self, tracker_data: Dict[str, Any]):
        """
        Prend en entrée le dictionnaire renvoyé par le tracker.
        Déplace les vidéos dans les bons dossiers selon le type de mouvement.
        """
        results = {
            "movement_detected": {},
            "immobile_detected": {}
        }

        for video_name, data in tracker_data.items():
            print(f"[Pipeline] Traitement de la vidéo : {video_name}")
            moved = False

            # Analyse du mouvement pour chaque classe dans simpletracking
            simpletracking = data.get("simpletracking", {})

            for cls, info in simpletracking.items():
                if info.get("detected", False):
                    results["movement_detected"].setdefault(video_name, []).append(cls)
                    folder_map = {
                        "scanner": "moving_scanners",
                        "plastic-bag": "moving_plastic_bag",
                        "rag": "moving_rag"
                    }
                    if cls in folder_map:
                        self.move_video(video_name, folder_map[cls])
                        moved = True
                else:
                    # Si pas de mouvement, considérer comme immobile
                    results["immobile_detected"].setdefault(video_name, []).append(cls)
                    folder_map = {
                        "scanner": "immobile_product",
                        "plastic-bag": "immobile_plastic_bag",
                        "rag": "immobile_rag"
                    }
                    if cls in folder_map:
                        self.move_video(video_name, folder_map[cls])

            if not moved and not simpletracking:
                print(f"[Pipeline] Aucun mouvement détecté pour {video_name}")

        return results

    def move_video(self, video_name: str, category: str):
        """Déplace une vidéo dans le sous-dossier correspondant"""
        src_path = os.path.join(self.video_dir, video_name)
        if category not in self.output_dir:
            print(f"[Pipeline] Catégorie inconnue : {category}")
            return

        dest_folder = self.output_dir[category]
        dest_path = os.path.join(dest_folder, os.path.basename(src_path))
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
            print(f"[Pipeline] Vidéo déplacée vers : {dest_path}")
        else:
            print(f"[Pipeline] Fichier non trouvé : {src_path}")


if __name__ == '__main__':
    tracker= UnifiedWatcher("config.yaml", detector=Detector("config.yaml"), video_path="videos/video_test_CarrefourSP_TBE1_20250924T155916_2.mp4")
    tracker_data= tracker.run_on_folder("videos", show_video=False)
    print(tracker_data)


    pipeline = Pipeline(video_dir='videos')
    results = pipeline.run(tracker_data)
    print(results)
