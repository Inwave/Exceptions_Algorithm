import time
from functools import wraps
from collections import defaultdict
import matplotlib.pyplot as plt

_PROFILE_DATA = defaultdict(list)

def profiled(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start
        if duration>0.005:
            _PROFILE_DATA[func.__qualname__].append(duration)
            
            #print(f"[Profiler] {func.__qualname__} took {duration:.4f}s")
        return result
    return wrapper




def plot_profile_stats(smoothing_window=1):
    """Affiche un graphique des temps d'exécution collectés"""
    if not _PROFILE_DATA:
        print("⚠️ Aucune donnée à afficher.")
        return

    plt.figure(figsize=(10, 6))
    for func_name, times in _PROFILE_DATA.items():
        if smoothing_window > 1:
            smoothed = [
                sum(times[max(0, i - smoothing_window + 1):i + 1]) / min(i + 1, smoothing_window)
                for i in range(len(times))
            ]
            y_vals = smoothed
        else:
            y_vals = times

        plt.plot(range(len(times)), y_vals, label=func_name)

    plt.xlabel("Appel #")
    plt.ylabel("Durée (s)")
    plt.title("Profiling des fonctions décorées")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()