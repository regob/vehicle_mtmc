from typing import List
import numpy as np
from mot.tracklet import Tracklet


class MulticamTracklet:
    """A union of (possibly multiple, but at least one) single camera tracklets."""

    def __init__(self, new_id: int, single_cam_tracks: List[Tracklet], n_cams: int):
        """
        Parameters
        ----------
        new_id: The global id of the multicam tracklet.
        single_cam_tracks: the single camera tracklets that form this multicam one.
        """
        self.id = new_id
        if len(single_cam_tracks) == 0:
            raise ValueError(
                "Invalid single_cam_tracks, at least one is needed.")
        self._tracks = single_cam_tracks
        self._mean_feature = None
        self._cams = None
        self._n_cams = n_cams

    def __hash__(self):
        return hash(tuple(t.id for t in self._tracks))

    def __eq__(self, other):
        return id(self) == id(other)

    @property
    def tracks(self):
        """Single cam tracklets contained."""
        return self._tracks

    @property
    def n_cams(self):
        """Total number of cameras in the system."""
        return self._n_cams

    @property
    def mean_feature(self):
        """Mean feature of all single cam tracklets."""
        if self._mean_feature is None:
            self._mean_feature = np.zeros_like(self._tracks[0].mean_feature)
            for track in self._tracks:
                self._mean_feature += track.mean_feature
            self._mean_feature /= np.linalg.norm(self._mean_feature)
        return self._mean_feature

    @property
    def cams(self):
        """Camera occurence bitmap."""
        if self._cams is None:
            self._cams = 0
            for track in self._tracks:
                self._cams |= 1 << track.cam
        return self._cams

    @property
    def inverse_cams(self):
        """Bitmap of cams not occuring in this tracklet if there are n_cams in total."""
        bmp = (1 << self._n_cams) - 1
        return bmp ^ self.cams

    def merge_with(self, other: 'MulticamTracklet'):
        """Merge an other multicam tracklet to this one."""
        self._tracks.extend(other.tracks)
        self._mean_feature = None
        if self._cams is not None:
            for track in other.tracks:
                self._cams |= 1 << track.cam

    def finalize(self):
        """Finalize single cam tracks contained in this mtrack (assign the same id to them)."""
        for track in self._tracks:
            track.track_id = self.id
        # TODO: make static attributes the same?


def get_tracks_by_cams(multicam_tracks: List[MulticamTracklet]) -> List[List[Tracklet]]:
    """Return multicam tracklets sorted by cameras."""
    if len(multicam_tracks) == 0:
        return []
    tracks_per_cam = [[] for _ in range(multicam_tracks[0].n_cams)]
    for mtrack in multicam_tracks:
        for track in mtrack.tracks:
            tracks_per_cam[track.cam].append(track)
    return tracks_per_cam


def have_mutual_cams(mtrack1: MulticamTracklet, mtrack2: MulticamTracklet) -> bool:
    """Checks whether two mutlicam tracklets share any cameras."""
    return bool(mtrack1.cams & mtrack2.cams)
