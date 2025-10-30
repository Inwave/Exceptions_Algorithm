import time
from functools import wraps
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

_PROFILE_DATA = defaultdict(list)

def profiled(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start
        if func.__qualname__ =='UnifiedWatcher.run':
            print(f"[Profiler] {func.__qualname__} took {duration:.4f}s")
        if func.__qualname__!= 'UnifiedWatcher.run':
            _PROFILE_DATA[func.__qualname__].append(duration)
        if duration>1: 
            print(f"[Profiler] {func.__qualname__} took {duration:.4f}s")
        return result
    return wrapper

def plot_profile_stats(smoothing_window=1):
    
    if not _PROFILE_DATA:
        print("[Profiler] No profiling data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Execution times ---
    ax = axes[0]
    for func_name, times in _PROFILE_DATA.items():
        if smoothing_window > 1:
            smoothed = [
                sum(times[max(0, i - smoothing_window + 1):i + 1]) / min(i + 1, smoothing_window)
                for i in range(len(times))
            ]
            y_vals = smoothed
        else:
            y_vals = times

        ax.plot(range(len(times)), y_vals, label=func_name)

    ax.set_xlabel("Call #")
    ax.set_ylabel("ExecutionTime (s)")
    ax.set_yscale("log")
    ax.set_title("Profiling of decorated functions")
    ax.legend()
    ax.grid(True)

    # --- Mean Times ---
    ax2 = axes[1]
    func_names = list(_PROFILE_DATA.keys())
    avg_times = [np.mean(times) for times in _PROFILE_DATA.values()]
    
    bars = ax2.bar(func_names, avg_times)
    ax2.set_yscale("log")
    ax2.set_ylabel("Mean execution time")
    ax2.set_title("Average execution times of decorated functions")
    ax2.grid(axis="y")

    
    for bar, avg in zip(bars, avg_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{avg:.2f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()