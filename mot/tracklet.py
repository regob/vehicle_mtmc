from scipy.cluster import vq
import numpy as np
from mot.attributes import STATIC_ATTRIBUTES, DYNAMIC_ATTRIBUTES


class Tracklet:
    """ The track of an object on the video. """

    def __init__(self, track_id):
        self.features = []
        self._mean_feature = None
        self.track_id = track_id

        # frame indices of the bounding boxes
        self.frames = []

        # bounding boxes in tlwh format
        self.bboxes = []

        # zone_id's for each bbox
        self.zones = []

        # confidence level for each bbox
        self.conf = []

        # static features of the track
        self.static_attributes = {}

        # dynamic attributes of the track
        self.dynamic_attributes = {}

        # global attributes in multi-camera systems, not used in MOT
        self.cam = None
        self.global_start, self.global_end = None, None

    def __repr__(self):
        return f"Tracklet(track_id={self.track_id}, num_frames: {len(self.frames)}, num_features:{len(self.features)})"

    def __hash__(self):
        return hash(self.track_id)

    @property
    def mean_feature(self):
        if self._mean_feature is None:
            self.compute_mean_feature()
        return self._mean_feature

    def update(self, frame_num, bbox, conf, feature=None, static_attributes=None, dynamic_attributes=None, zone_id=None):
        """Add a new detection to the track."""
        if feature is not None:
            self.features.append(feature)
        self.frames.append(frame_num)
        self.bboxes.append(bbox)
        self.conf.append(conf)
        if static_attributes:
            for k, v in static_attributes.items():
                self.static_attributes.setdefault(k, []).append(v)
        if dynamic_attributes:
            for k, v in dynamic_attributes.items():
                self.dynamic_attributes.setdefault(k, []).append(v)
        if zone_id is not None:
            self.zones.append(zone_id)

    def compute_mean_feature(self, method="area_avg"):
        """Compute a single feature from the frame-by-frame features to describe the track.

        Parameters
        ----------
        method: str
            Method to use from ('area_avg', 'mean').
            area_avg: sum the features multiplied by the area of the bounding box, then divide the result
            by the sum of areas.
            mean: take the unweighted mean of the features.

        Returns
        -------
        mean_feature: np.array
        """
        self._mean_feature = np.zeros_like(self.features[0])
        if method == "area_avg":
            div = min(map(lambda x: x[2] * x[3], self.bboxes))
        for i, f in enumerate(self.features):
            if method == "area_avg":
                area = self.bboxes[i][2] * self.bboxes[i][3]
                self._mean_feature += f * (area / div)
            else:
                self._mean_feature += f

        norm = np.linalg.norm(self._mean_feature)
        self._mean_feature = self._mean_feature / norm
        return self._mean_feature

    def cluster_features(self, k):
        """Reduce the re-id features by K-means clustering."""
        if len(self.features) <= k:
            return

        f = np.array(self.features)
        centroids = vq.kmeans(f, k)[0]
        self.features = [feature for feature in centroids]
        return self.features

    def predict_final_static_attributes(self):
        """Update the static attributes to describe the whole track instead of frame-by-frame values."""
        static_f = {}
        for k, v in self.static_attributes.items():
            if isinstance(v, int):
                # the attributes are alredy finalized
                return
            preds = np.zeros((len(STATIC_ATTRIBUTES[k]), ))

            # there is an attribute for each frame (this is preferred)
            if len(v) == len(self.bboxes):
                for pred, bbox in zip(v, self.bboxes):
                    preds[pred] += bbox[2] * bbox[3]
            else:
                for pred in v:
                    preds[pred] += 1

            static_f[k] = int(preds.argmax())
        self.static_attributes = static_f
        return static_f

    def finalize_speed(self, mean_mul=2.0, window_size=5, max_speed=180):
        """Refines per-frame speed values."""
        if "speed" not in self.dynamic_attributes:
            return
        speeds = self.dynamic_attributes["speed"]
        if len(speeds) == 0:
            return

        # set the first few missing measurements to the first non-missing one
        for i in range(len(speeds)):
            if speeds[i] < 0 or speeds[i] > max_speed:
                continue
            speeds[:i] = [speeds[i]] * i
            break
        start = i

        # set the last few missing measurements to the last non-missing one
        for i in reversed(range(len(speeds))):
            if speeds[i] < 0 or speeds[i] > max_speed:
                continue
            speeds[i:] = [speeds[i]] * (len(speeds) - i)
            break
        end = i

        # fill missing values forward
        fw_fill = []
        for i in range(start, end + 1):
            if speeds[i] < 0 or speeds[i] > max_speed:
                fw_fill.append(fw_fill[-1])
            else:
                fw_fill.append(speeds[i])

        # fill missing values backward
        bw_fill = []
        for i in range(end, start - 1, -1):
            if speeds[i] < 0 or speeds[i] > max_speed:
                bw_fill.append(bw_fill[-1])
            else:
                bw_fill.append(speeds[i])
        bw_fill.reverse()

        # set the [start, end] range to the mean of the fw and the bw filled values
        if end > start:
            speeds[start:end+1] = [int((x + y)/2) for (x,y) in zip(fw_fill, bw_fill)]


        # smoothen values with a sliding window
        for i in range(len(speeds)):
            l = max(0, i - window_size // 2)
            r = min(len(speeds) - 1, i + window_size // 2)
            speeds[i] = sum(speeds[l:r+1]) // (r - l + 1)


    def zone_enter_leave_frames(self, zone_id):
        """Frame indices when the track entered and left a given zone."""
        enter, leave = -1, -1
        for fr, z in zip(self.frames, self.zones):
            if z == zone_id:
                if enter < 0:
                    enter = fr
                leave = fr
        return enter, leave
