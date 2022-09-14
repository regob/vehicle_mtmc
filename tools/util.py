import time
from collections import deque


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
    t = time.time()

    @classmethod
    def start():
        Timer.t = time.time()

    @classmethod
    def show(callname=""):
        print("Call ({}) took {:.5f} seconds.".format(
            callname, time.time() - Timer.t))
        Timer.t = time.time()

