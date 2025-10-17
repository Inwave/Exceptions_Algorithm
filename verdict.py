import yaml

class Verdict:
    """
    Analyze results of movements detection and determine if there's sginificant movement
    for Non-Products Objects in the ROIs.
    
    """

    def __init__(self, config_path):
        """
        Args:
            movement_data (dict): Dict with details of the differents movement detected by the SimpleTracker class
        """
        config=self.load_config(config_path)
        self.min_duration = config.get("min_duration", 0.5) #min_duration
        self.min_movement_count = config.get("min_movement_count", 1)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get('verdict', {})
    

    def analyze(self, movement_data) -> dict:
        """
        Analyze movements of each class and return global verdict
        """
        result = {
            "significant_movement": False,
            "classes_moved": [],
            "details": {}
        }

        self.data = movement_data
        for cls, info in self.data.items():
            significant_movements = [
                m for m in info.get("movements", [])
                if m["duration_s"] >= self.min_duration
            ]

            count = len(significant_movements)
            total_duration = sum(m["duration_s"] for m in significant_movements)

            result["details"][cls] = {
                "count": count,
                "total_duration": round(total_duration, 2),
                "detected": count > 0
            }

            if count >= self.min_movement_count:
                result["classes_moved"].append(cls)

        
        result["significant_movement"] = len(result["classes_moved"]) > 0

        return result
    
    def alert_message(self, class_list):
        return f"Significant movement of Non-Products Objects detected in ROI for classes: {', '.join(class_list)}"

    
    def verdict_message(self, movement_data):
        results=self.analyze(movement_data)
        if results["significant_movement"]:
            return self.alert_message(results["classes_moved"])
        else:
            return "No significant movement of Non-Products Objects detected, might be a positive alert"

            
    def verdcit(self):
        results=self.analyse()    
        return {'significant_movement':results['significant_movement'], 'verdict_message':self.verdict_message()} 
    

