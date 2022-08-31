from typing import List, Union, Dict
import numpy as np

from mot.detection import Detection
from mot.tracklet import Tracklet
from mot.zones import ZoneMatcher
from mot.deep_sort import preprocessing, nn_matching
from mot.deep_sort.tracker import Tracker


class TrackerBase:

    def __init__(self, zone_matcher: Union[ZoneMatcher, None] = None):
        self._tracks = {}
        self._active_track_ids = set()
        self._zone_matcher = zone_matcher

    @property
    def tracks(self):
        return self._tracks

    @property
    def active_track_ids(self):
        return self._active_track_ids

    def update(self, frame_num: int, detections: List[Detection], static_attributes: Union[Dict, None] = None,
               dynamic_attributes: Union[Dict, None] = None):
        raise NotImplementedError()

    def active_tracks(self):
        return [self._tracks[i] for i in self._active_track_ids]


class DeepsortTracker(TrackerBase):

    def __init__(self, metric: str,
                 max_dist: float,
                 nn_budget: int,
                 n_init: int,
                 max_iou_dist: float = 0.7,
                 max_age: int = 60,
                 zone_matcher: Union[ZoneMatcher, None] = None):
        """Initialize a DeepSORT tracker.

        Parameters
        ----------
        metric: 'cosine' or 'euclidean'
        max_dist:
        """
        super().__init__(zone_matcher)
        self._metric = nn_matching.NearestNeighborDistanceMetric(
            metric, max_dist, nn_budget)
        self._tracker = Tracker(self._metric, max_iou_dist, max_age, n_init)

    def update(self,
               frame_num: int,
               detections: List[Detection],
               static_attributes: Union[Dict, None] = None,
               dynamic_attributes: Union[Dict, None] = None):
        """Update the tracker with detections from a new frame."""

        # monkey patch attributes to detections to make it easier 
        # when retrieving the tracks after the update
        for i, det in enumerate(detections):
            if static_attributes:
                det.static_attributes = {k: static_attributes[k][i] for k in static_attributes}
            if dynamic_attributes:
                det.dynamic_attributes = {k: dynamic_attributes[k][i] for k in dynamic_attributes}
                
        self._tracker.predict()
        self._tracker.update(detections)

        self._active_track_ids = set()
        for track in self._tracker.tracks:
            if track.track_id not in self._tracks:
                self._tracks[track.track_id] = Tracklet(track.track_id)
            if track.time_since_update > 1:
                continue
            
            self._active_track_ids.add(track.track_id)
            tracklet = self._tracks[track.track_id]
            det = track.last_detection
            cx, cy = int(det.tlwh[0] + det.tlwh[2] / 2), int(det.tlwh[1] + det.tlwh[3] / 2)
            zone_id = self._zone_matcher.find_zone_for_point(cx, cy) if self._zone_matcher is not None else None
            tracklet.update(frame_num, det.tlwh, det.confidence, det.feature, det.static_attributes,
                            det.dynamic_attributes, zone_id)
            
            
            
        
        
