from scipy.cluster import vq
import numpy as np


class Tracklet:
    """ The track of an object on the video. """

    def __init__(self, track_id):
        self.features = []
        self.frames = []
        self.track_id = track_id

        # bounding boxes in tlwh format
        self.bboxes = []

        # zone_id's for each bbox
        self.zones = []

        self.static_features = {}

    def update(self, frame_num, bbox, feature, static_features=None, zone_id=None):
        self.features.append(feature)
        self.frames.append(frame_num)
        self.bboxes.append(bbox)
        if static_features:
            for k, v in static_features.items():
                self.static_features.setdefault(k, []).append(v)
        if zone_id is not None:
            self.zones.append(zone_id)

    def cluster_features(self, k):
        if len(self.features) <= k:
            return

        f = np.array(self.features)
        centroids = vq.kmeans(f, k)[0]
        self.features = [feature for feature in centroids]