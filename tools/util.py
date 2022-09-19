import time
import argparse
from collections import deque
import numpy as np


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", help="config yaml file")
    parser.add_argument("--log_level", default="info", help="logging level")
    parser.add_argument("--log_filename", default="log.txt",
                        help="log file under output dir")
    parser.add_argument("--no_log_stdout", action="store_true",
                        help="do not log to stdout")
    return parser.parse_args()


class FrameRateCounter:
    def __init__(self, window=5):
        self.timestamps = deque()
        self.window = window

    def step(self):
        self.timestamps.append(time.time())

    def value(self):
        now = time.time()
        while len(self.timestamps) > 0 and (now - self.timestamps[0] > self.window):
            self.timestamps.popleft()
        return len(self.timestamps) / self.window


class Timer:

    def __init__(self):
        self.t = time.time()

    def start(self):
        self.t = time.time()

    def elapsed(self):
        return time.time() - self.t

    def fetch_restart(self):
        diff = self.elapsed()
        self.start()
        return diff

    def print_restart(self, callname=""):
        print("Call ({}) took {:.5f} seconds.".format(
            callname, time.time() - self.t))
        self.t = time.time()


class Benchmark:

    def __init__(self):
        self.timer = Timer()
        self.calls = {}

    def restart_timer(self):
        self.timer.start()

    def register_call(self, callname):
        self.calls.setdefault(callname, []).append(self.timer.fetch_restart())

    def reset(self):
        self.timer.start()
        self.calls = {}

    def get_benchmark(self, unit="ms", precision=4):
        multip = 1000 if unit == "ms" else 1
        mxwidth = [0] * 4
        records = [["name", "min", "mean", "max"]]
        for k, times in self.calls.items():
            times = np.array(times) * multip
            record = [k]
            for x in [times.min(), times.mean(), times.max()]:
                record.append(str(round(x, precision)))
            records.append(record)
            for i, x in enumerate(record):
                mxwidth[i] = max(mxwidth[i], len(x))
        lines = []
        for record in records:
            line = f"{record[0].ljust(mxwidth[0])} " + " ".join([x.center(w)
                                                                 for x, w in zip(record[1:], mxwidth[1:])])
            lines.append(line)
        sepline = "-" * (sum(mxwidth) + 4)
        return "\n".join([lines[0], sepline] + lines[1:] + [sepline])
