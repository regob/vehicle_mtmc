import numpy as np


def iou(box1_tlwh, box2_tlwh):
    box1, box2 = np.array(box1_tlwh), np.array(box2_tlwh)
    box1[2:] += box1[:2]
    box2[2:] += box2[:2]

    x = min(box1[2], box2[2]) - max(box1[0], box2[0])
    y = min(box1[3], box2[3]) - max(box1[1], box2[1])

    if x > 0 and y > 0:
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box1[3] - box2[1])
        return (x * y) / (a1 + a2 - x * y)
    return .0


def euclidean_dist(v1: np.ndarray, v2: np.ndarray):
    return np.linalg.norm(v1 - v2)

def cosine_sim(v1: np.ndarray, v2: np.ndarray, already_normed=False):
    if already_normed:
        return np.dot(v1, v2)
    return np.dot(v1, v2) / np.linalg.norm(v1, 2) / np.linalg.norm(v2, 2)


