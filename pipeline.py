from detection_analysis import Detector
from simple_tracking import SimpleTracking
from verdict import Verdict

import shutil
import os
from typing import Dict, Any

class Pipeline:
    def __init__(self, config_path, video_dir):
        self.config_path = config_path
        self.detector = Detector(config_path=config_path)
        self.verdict = Verdict(config_path=config_path)
        self.video_dir=video_dir
        self.output_dir={
            "real_alerts": os.path.join(video_dir, "real_alerts"),
            "scanner_alerts": os.path.join(video_dir, "scanner_alerts"),
            "rag_alerts": os.path.join(video_dir, "rag_alerts"),
            "plastic_bags_alerts": os.path.join(video_dir, "plastic_bags_alerts"),
        }

        # Création des sous-dossiers si besoin
        for folder in self.output_dir.values():
            os.makedirs(folder, exist_ok=True)

        

    def run(self):  



        results = {
            "movement_detected": {},
            "potential_real_alerts": []
        }

        # Lister toutes les vidéos
        video_files = [
            f for f in os.listdir(self.video_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]

        if not video_files:
            print("[Pipeline] No file found in :", self.video_dir)
            return results

        for video_name in video_files:
            video_path = os.path.join(self.video_dir, video_name)
            print(f"[Pipeline] Traitement de la vidéo : {video_name}")

            tracker = SimpleTracking(config_path=self.config_path, video_path=video_path, detector=self.detector)
            

            
            movement_data = tracker.run(show_video=False, save_path=None, verbose=False)
            print('movement data:', movement_data)
            
            analysis = self.verdict.analyze(movement_data=movement_data)
            print(self.verdict.verdict_message(movement_data))

            # Save result
            if analysis["significant_movement"]:
                results["movement_detected"][video_name] = analysis['classes_moved']
                if "scanner" in analysis.get("classes_moved", []):
                    self.move_video(video_path, "scanner_alerts")
                if 'rag' in analysis.get("classes_moved", []):
                    self.move_video(video_path, "rag_alerts")
                if 'plastic-bags' in analysis.get("classes_moved", []):
                    self.move_video(video_path, "plastic_bags_alerts")
            else:
                results["potential_real_alerts"].append(video_name)
                self.move_video(video_path, "real_alerts")

        return results
    
    def move_video(self, src_path: str, category: str):
        """Déplace une vidéo dans le sous-dossier correspondant"""
        if category not in self.output_dir:
            print(f"[Pipeline] Catégorie inconnue : {category}")
            return

        dest_folder = self.output_dir[category]
        dest_path = os.path.join(dest_folder, os.path.basename(src_path))
        shutil.move(src_path, dest_path)
        print(f"[Pipeline] Vidéo déplacée vers : {dest_path}")
    

if __name__=='__main__':
    pipeline = Pipeline(config_path='config.yaml', video_dir='videos')
    results = pipeline.run()
    print(results)