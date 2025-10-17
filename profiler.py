import time
from functools import wraps

def profiled(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start
        if duration>0.005:
            print(f"[Profiler] {func.__qualname__} took {duration:.4f}s")
        return result
    return wrapper