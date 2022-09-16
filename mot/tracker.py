"""Interfaces to the different trackers."""

from typing import List, Union, Dict, Set
import numpy as np

from detection.detection import Detection
from mot.tracklet import Tracklet
from mot.zones import ZoneMatcher
from mot.deep_sort import preprocessing, nn_matching
from mot.deep_sort.tracker import Tracker
from mot.byte_track.byte_tracker import BYTETracker, STrack
from tools import log


class TrackerBase:

    def __init__(self, zone_matcher: Union[ZoneMatcher, None] = None):
        self._tracks = {}
        self._active_track_ids = set()
        self._zone_matcher = zone_matcher

    @property
    def tracks(self) -> Dict[int, Tracklet]:
        """Dictionary of tracks keyed by ids."""
        return self._tracks

    @property
    def active_track_ids(self) -> Set[int]:
        """Set of active track ids."""
        return self._active_track_ids

    @property
    def active_tracks(self) -> List[Tracklet]:
        """List of active track objects."""
        return [self._tracks[i] for i in self._active_track_ids]

    def update(self, frame_num: int, detections: List[Detection], static_attributes: Union[Dict, None] = None,
               dynamic_attributes: Union[Dict, None] = None):
        raise NotImplementedError()


def monkey_patch_detections(detections: List[Detection],
                            static_attributes: Union[Dict, None] = None,
                            dynamic_attributes: Union[Dict, None] = None):
    """monkey patch attributes to detections."""
    for i, det in enumerate(detections):
        if static_attributes:
            det.static_attributes = {
                k: static_attributes[k][i] for k in static_attributes}
        else:
            det.static_attributes = {}
        if dynamic_attributes:
            det.dynamic_attributes = {
                k: dynamic_attributes[k][i] for k in dynamic_attributes}
        else:
            det.dynamic_attributes = {}
    return detections


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

        detections = monkey_patch_detections(
            detections, static_attributes, dynamic_attributes)
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
            cx, cy = int(det.tlwh[0] + det.tlwh[2] /
                         2), int(det.tlwh[1] + det.tlwh[3] / 2)
            zone_id = self._zone_matcher.find_zone_for_point(
                cx, cy) if self._zone_matcher else None
            tracklet.update(frame_num, det.tlwh, det.confidence, det.feature, det.static_attributes,
                            det.dynamic_attributes, zone_id)


class ByteTrackerOpts:
    def __init__(self, track_conf_thresh, new_track_conf_thresh, track_match_thresh, lost_track_keep_seconds):
        self.track_thresh = track_conf_thresh
        self.det_thresh = new_track_conf_thresh
        self.track_buffer = 30 * lost_track_keep_seconds
        self.match_thresh = track_match_thresh
        self.mot20 = False


class ByteTrackerIOU(TrackerBase):
    def __init__(self,
                 frame_rate=30,
                 track_conf_thresh=0.5,
                 new_track_conf_thresh=0.4,
                 track_match_thresh=0.8,
                 lost_track_keep_seconds=3,
                 zone_matcher: Union[ZoneMatcher, None] = None,
                 ):
        super().__init__(zone_matcher)
        byte_track_opts = ByteTrackerOpts(track_conf_thresh, new_track_conf_thresh,
                                          track_match_thresh, lost_track_keep_seconds)
        self._tracker = BYTETracker(
            args=byte_track_opts, frame_rate=frame_rate)

    def update(self,
               frame_num: int,
               detections: List[Detection],
               static_attributes: Union[Dict, None] = None,
               dynamic_attributes: Union[Dict, None] = None):
        """Update the tracker with detections from a new frame."""

        detections = monkey_patch_detections(
            detections, static_attributes, dynamic_attributes)

        # create input for bytetrack in the form of tlbr + score records
        byte_input = np.zeros((len(detections), 5), np.float)
        for i, det in enumerate(detections):
            byte_input[i, :4] = det.to_tlbr()
            byte_input[i, 4] = det.confidence

        stracks = self._tracker.update(byte_input)
        log.debug(f"Detections: {len(detections)}, active tracks: {len(stracks)}.")

        self._active_track_ids = set()
        for strack in stracks:
            if strack.track_id not in self._tracks:
                self._tracks[strack.track_id] = Tracklet(strack.track_id)
            self._active_track_ids.add(strack.track_id)
            track = self._tracks[strack.track_id]

            det = detections[strack.last_det_idx]
            cx, cy = int(det.tlwh[0] + det.tlwh[2] /
                         2), int(det.tlwh[1] + det.tlwh[3] / 2)
            zone_id = self._zone_matcher.find_zone_for_point(
                cx, cy) if self._zone_matcher else None
            track.update(frame_num, det.tlwh, det.confidence, det.feature, det.static_attributes,
                         det.dynamic_attributes, zone_id)
