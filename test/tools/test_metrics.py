import math
import numpy as np
from tools.metrics import iou, euclidean_dist, cosine_sim

EPS = 1e6

def test_iou_full():
    box1 = [0, 10, 200, 200]
    box2 = [100, 100, 50, 50]
    iou_score = iou(box1, box2)
    assert abs(iou_score - (50 * 50 / (200 * 200))) < EPS

def test_iou_none():
    box1 = [54, 39, 100, 78]
    box2 = [164, 10, 30, 30]
    assert iou(box1, box2) == 0.0

def test_iou_some():
    box1 = [50, 60, 100, 50]
    box2 = [80, 80, 110, 120]
    score = iou(box1, box2)
    exp_score = 70 * 30 / (100 * 50 + 110 * 120 - 70 * 30)
    assert abs(score - exp_score) < EPS

def test_euclidean():
    v1 = np.array([17, -4, -3])
    v2 = np.array([1, -5, 6])
    exp_score = math.sqrt(16 ** 2 + 1 + 9 ** 2)
    assert abs(euclidean_dist(v1, v2) - exp_score) < EPS

def test_cosine():
    v1 = np.array([0, -4])
    v2 = np.array([4, -4])
    exp_score = math.sqrt(2) / 2
    assert abs(cosine_sim(v1, v2) - exp_score) < EPS
