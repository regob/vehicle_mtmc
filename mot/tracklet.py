from scipy.cluster import vq
import numpy as np
import pandas as pd
import pickle


class Tracklet:
    def __init__(self, track_id):
        self.features = []
        self.frames = []
        self.track_id = track_id

        # bounding boxes in tlwh format
        self.bboxes = []
        self.static_features = {}

    def update(self, frame_num, bbox, feature, static_features=None):
        self.features.append(feature)
        self.frames.append(frame_num)
        self.bboxes.append(bbox)
        if static_features:
            for k, v in static_features.items():
                self.static_features.setdefault(k, []).append(v)

    def cluster_features(self, k):
        if len(self.features) <= k:
            return

        f = np.array(self.features)
        centroids = vq.kmeans(f, k)[0]
        self.features = [feature for feature in centroids]


def save_tracklets(tracklets, path, max_features=None):
    """Saves tracklets using pickle (with re-id features)"""
    if max_features is not None:
        for tracklet in tracklets:
            tracklet.cluster_features(max_features)
    with open(path, "wb") as fp:
        pickle.dump(tracklets, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_tracklets(pickled_path):
    """Loads a pickled list of tracklets."""
    with open(pickled_path, "rb") as f:
        tracklets = pickle.load(f)
    return tracklets


def save_tracklets_csv(tracklets, path):
    res = {
        "frame": [],
        "bbox_topleft_x": [],
        "bbox_topleft_y": [],
        "bbox_width": [],
        "bbox_height": [],
        "track_id": [],
    }
    for k in tracklets[0].static_features:
        res[k] = []

    for tracklet in tracklets:
        res["frame"].extend(tracklet.frames)
        for x, y, w, h in tracklet.bboxes:
            res["bbox_topleft_x"].append(x)
            res["bbox_topleft_y"].append(y)
            res["bbox_width"].append(w)
            res["bbox_height"].append(h)
        res["track_id"].extend([tracklet.track_id] * len(tracklet.frames))
        for static_f, values in tracklet.static_features:
            res[static_f].extend(values)

    # all columns should have the same length
    lengths = list(map(lambda col: len(col), res.values()))
    lengths_equal = list(map(lambda l: l == lengths[0], lengths))
    if not all(lengths_equal):
        for k, v in res.items():
            print(f"Items in column {k}: {len(v)}")
        raise ValueError("Error: not all column lengths are equal.")

    df = pd.DataFrame(res)
    df.to_csv(path, index=False)
