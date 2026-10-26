import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


class timer:
    def __init__(self, what):
        self.what = what

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        print(f"{self.what}: {time.perf_counter() - self.t0:.2f} s", flush=True)
