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

    # output
    # def compute_mean_feature(self, method="area_avg"):
    #     """Compute a single feature from the frame-by-frame features to describe the track.

    #     Parameters
    #     ----------
    #     method: str
    #         Method to use from ('area_avg', 'mean').
    #         area_avg: sum the features multiplied by the area of the bounding box, then divide the result
    #         by the sum of areas.
    #         mean: take the unweighted mean of the features.

    #     Returns
    #     -------
    #     mean_feature: np.array
    #     """
    #     if len(self.features) == 0:
    #         return self._mean_feature
    #     self._mean_feature = np.zeros_like(self.features[0])
    #     if method == "area_avg":
    #         div = min(map(lambda x: x[2] * x[3], self.bboxes))
    #     for i, f in enumerate(self.features):
    #         if method == "area_avg":
    #             area = self.bboxes[i][2] * self.bboxes[i][3]
    #             self._mean_feature += f * (area / div)
    #         else:
    #             self._mean_feature += f

    #     norm = np.linalg.norm(self._mean_feature)
    #     self._mean_feature = self._mean_feature / norm
    #     return self._mean_feature

    # # output3
    # def compute_mean_feature(self, method="adaptive_weighted"):
    #     if len(self.features) == 0:
    #         return self._mean_feature

    #     features = np.array(self.features)
    #     confs = np.array(self.conf)  # 检测置信度
    #     areas = np.array([w*h for (_, _, w, h) in self.bboxes])  # 框面积

    #     # 1. 归一化置信度和面积
    #     confs_norm = (confs - confs.min()) / (confs.max() - confs.min() + 1e-6)
    #     areas_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1e-6)

    #     # 2. 计算帧质量分数（可加权组合）
    #     quality = 0.7 * confs_norm + 0.3 * areas_norm

    #     # 3. 动态阈值筛选（如只用前80%高分帧）
    #     thresh = np.percentile(quality, 20)  # 只用高于20分位的帧
    #     mask = quality > thresh
    #     features_sel = features[mask]
    #     quality_sel = quality[mask]

    #     # 4. 自适应加权融合
    #     weights = quality_sel / (quality_sel.sum() + 1e-6)
    #     mean_feature = np.sum(features_sel * weights[:, None], axis=0)
    #     mean_feature = mean_feature / (np.linalg.norm(mean_feature) + 1e-6)
    #     self._mean_feature = mean_feature
    #     return self._mean_feature

    # # output4
    # def compute_mean_feature(self, method="adaptive_weighted"):
    #     if len(self.features) == 0:
    #         return self._mean_feature

    #     features = np.array(self.features)
    #     confs = np.array(self.conf)
    #     areas = np.array([w*h for (_, _, w, h) in self.bboxes])

    #     # 归一化
    #     confs_norm = (confs - confs.min()) / (confs.max() - confs.min() + 1e-6)
    #     areas_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1e-6)
    #     quality = 0.7 * confs_norm + 0.3 * areas_norm

    #     # 不做mask，直接加权平均
    #     weights = quality / (quality.sum() + 1e-6)
    #     mean_feature = np.sum(features * weights[:, None], axis=0)
    #     mean_feature = mean_feature / (np.linalg.norm(mean_feature) + 1e-6)
    #     self._mean_feature = mean_feature
    #     return self._mean_feature

    # # # output5
    # def compute_mean_feature(self, method="hybrid_adaptive"):
    #     if len(self.features) == 0:
    #         return self._mean_feature

    #     features = np.array(self.features)
    #     if method == "hybrid_adaptive":
    #         confs = np.array(self.conf)
    #         areas = np.array([w*h for (_, _, w, h) in self.bboxes])
    #         # 1. 归一化置信度和面积
    #         confs_norm = (confs - confs.min()) / (confs.max() - confs.min() + 1e-6)
    #         areas_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1e-6)
    #         # 2. 初步均值特征
    #         mean_feat0 = features.mean(axis=0)
    #         mean_feat0 = mean_feat0 / (np.linalg.norm(mean_feat0) + 1e-6)
    #         # 3. 每帧与均值特征的余弦相似度
    #         feats_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
    #         sims = feats_norm @ mean_feat0
    #         sims_norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-6)
    #         # 4. 帧间特征距离
    #         dists = np.linalg.norm(features[1:] - features[:-1], axis=1)
    #         dist_mean, dist_std = dists.mean(), dists.std()
    #         abnormal_idx = np.where(dists > dist_mean + 2 * dist_std)[0] + 1
    #         mask_dist = np.ones(len(features), dtype=bool)
    #         mask_dist[abnormal_idx] = False
    #         # 5. 多源分数融合
    #         quality = 0.4 * confs_norm + 0.2 * areas_norm + 0.4 * sims_norm
    #         # 剔除低于分数阈值的帧（如20分位）
    #         thresh = np.percentile(quality, 20)
    #         mask_quality = quality > thresh
    #         # 合并mask
    #         mask = mask_quality & mask_dist
    #         features_sel = features[mask]
    #         quality_sel = quality[mask]
    #         # 6. 滑动窗口时序平滑
    #         window_size = 5
    #         smoothed_features = np.copy(features_sel)
    #         for i in range(len(features_sel)):
    #             l = max(0, i - window_size // 2)
    #             r = min(len(features_sel), i + window_size // 2 + 1)
    #             smoothed_features[i] = features_sel[l:r].mean(axis=0)
    #         # 7. 多源分数加权融合
    #         weights = quality_sel / (quality_sel.sum() + 1e-6)
    #         mean_feature = np.sum(smoothed_features * weights[:, None], axis=0)
    #         mean_feature = mean_feature / (np.linalg.norm(mean_feature) + 1e-6)
    #         self._mean_feature = mean_feature
    #         return self._mean_feature
    #     # 保持原有逻辑
    #     if method == "mean":
    #         self._mean_feature = np.mean(self.features, axis=0)
    #         self._mean_feature = self._mean_feature / (np.linalg.norm(self._mean_feature) + 1e-6)
    #         return self._mean_feature
    #     # 默认area_avg
    #     self._mean_feature = np.zeros_like(self.features[0])
    #     div = min(map(lambda x: x[2] * x[3], self.bboxes))
    #     for i, f in enumerate(self.features):
    #         area = self.bboxes[i][2] * self.bboxes[i][3]
    #         self._mean_feature += f * (area / div)
    #     norm = np.linalg.norm(self._mean_feature)
    #     self._mean_feature = self._mean_feature / (norm + 1e-6)
    #     return self._mean_feature

    # output6
    def compute_mean_feature(self, method="hybrid_adaptive"):
        if len(self.features) == 0:
            return self._mean_feature

        features = np.array(self.features)
        confs = np.array(self.conf)
        areas = np.array([w*h for (_, _, w, h) in self.bboxes])
        confs_norm = (confs - confs.min()) / (confs.max() - confs.min() + 1e-6)
        areas_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1e-6)
        mean_feat0 = features.mean(axis=0)
        mean_feat0 = mean_feat0 / (np.linalg.norm(mean_feat0) + 1e-6)
        feats_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
        sims = feats_norm @ mean_feat0
        sims_norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-6)

        # 只用置信度加权
        # quality = confs_norm
        # 只用特征一致性加权
        # quality = sims_norm
        # 多源加权
        quality = 0.4 * confs_norm + 0.2 * areas_norm + 0.4 * sims_norm

        # 不做剔除，直接加权平均
        import scipy.special
        weights = scipy.special.softmax(quality)
        mean_feature = np.sum(features * weights[:, None], axis=0)
        mean_feature = mean_feature / (np.linalg.norm(mean_feature) + 1e-6)
        self._mean_feature = mean_feature
        return self._mean_feature

    # def compute_mean_feature(self, method="area_avg"):
    
    #     if len(self.features) == 0:
    #         return self._mean_feature
    #     # 1. 原始ReID特征
    #     reid_feat = np.zeros_like(self.features[0])
    #     if method == "area_avg":
    #         div = min(map(lambda x: x[2] * x[3], self.bboxes))
    #     for i, f in enumerate(self.features):
    #         if method == "area_avg":
    #             area = self.bboxes[i][2] * self.bboxes[i][3]
    #             reid_feat += f * (area / div)
    #         else:
    #             reid_feat += f
    #     reid_feat = reid_feat / (np.linalg.norm(reid_feat) + 1e-6)

    #     # print(f"[DEBUG] Track {self.track_id} static_attributes:", self.static_attributes)

    #     # 2. 属性特征（如颜色、车型 one-hot 或概率分布）
    #     attr_feat = []
    #     if hasattr(self, "static_attributes") and self.static_attributes:
    #         for k, v in self.static_attributes.items():
    #             if isinstance(v, int):
    #                 one_hot = np.zeros(len(STATIC_ATTRIBUTES[k]))
    #                 one_hot[v] = 1
    #                 attr_feat.append(one_hot)
    #     if attr_feat:
    #         attr_feat = np.concatenate(attr_feat)
    #     else:
    #         attr_feat = np.zeros(0)
        
    #     # print(f"[DEBUG] Track {self.track_id} attr_feat:", attr_feat)

    #     # print(f"[DEBUG] Track {self.track_id} dynamic_attributes:", self.dynamic_attributes)

    #     # # 3. 运动特征（如平均速度、方向等）
    #     # motion_feat = []
    #     # # if hasattr(self, "dynamic_attributes") and "speed" in self.dynamic_attributes:
    #     # #     speed = np.array(self.dynamic_attributes["speed"])
    #     # #     motion_feat = np.array([np.mean(speed), np.std(speed)])
    #     # # if motion_feat is not None and len(motion_feat) > 0:
    #     # #     motion_feat = np.array(motion_feat)
    #     # # else:
    #     # #     motion_feat = np.zeros(2)
    #     # # 异常值剔除（阈值法）
    #     # if hasattr(self, "dynamic_attributes") and "speed" in self.dynamic_attributes:
    #     #     speed = np.array(self.dynamic_attributes["speed"])
    #     #     # 异常值剔除（阈值法）
    #     #     valid_speed = speed[(speed > 0) & (speed < 120)]
    #     #     if len(valid_speed) == 0:
    #     #         mean_speed, std_speed = 0, 0
    #     #     else:
    #     #         mean_speed = np.mean(valid_speed) / 120.0  # 归一化
    #     #         std_speed = np.std(valid_speed) / 120.0
    #     #     motion_feat = np.array([mean_speed, std_speed])
    #     # else:
    #     #     motion_feat = np.zeros(2)

    #     # print(f"[DEBUG] Track {self.track_id} motion_feat:", motion_feat)

    #     # 4. 融合
    #     fused = np.concatenate([reid_feat, attr_feat])
    #     self._mean_feature = fused / (np.linalg.norm(fused) + 1e-6)
    #     return self._mean_feature

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
            speeds[start:end+1] = [int((x + y)/2) for (x, y) in zip(fw_fill, bw_fill)]

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
