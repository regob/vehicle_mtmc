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
            raise ValueError("Invalid single_cam_tracks, at least one is needed.")
        self._tracks = single_cam_tracks
        self._mean_feature = None
        self._cams = None
        self._n_cams = n_cams


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
                self._cams |= track.cam

