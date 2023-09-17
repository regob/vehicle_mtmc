from typing import List, Union
import time
import heapq
import numpy as np
import torch
from scipy.cluster import hierarchy

from tools.metrics import cosine_sim
from tools.data_structures import DSU
from tools import log
from mot.tracklet import Tracklet
from mtmc.cameras import CameraLayout
from mtmc.multicam_tracklet import MulticamTracklet, have_mutual_cams

def tracks_compatible(track1: Tracklet, track2: Tracklet, cams: Union[CameraLayout, None]) -> bool:
    """Check whether two tracks can be connected to each other."""
    cam1, cam2 = track1.cam, track2.cam
    # if there is no cam layout, we only check if the tracks are on the same camera
    if cams is None:
        return cam1 != cam2
    # same camera -> they cannot be connected
    if cam1 == cam2:
        return False

    t1_start, t1_end = track1.global_start, track1.global_end
    t2_start, t2_end = track2.global_start, track2.global_end

    # is track1 -> track2 transition possible?
    # for this, [t2_start, t2_end] has to intersect with interval I = [t1_end + dtmin, t1_end + dtmax]
    # that is: track2 starts before I ends and I starts before track2 ends
    if (cams.cam_compatibility_bitmap(cam1) & (1 << cam2) > 0) and \
       t2_start <= t1_end + cams.dtmax[cam1][cam2] and \
       t1_end + cams.dtmin[cam1][cam2] <= t2_end:
        return True

    # check the track2 -> track1 transition too
    if (cams.cam_compatibility_bitmap(cam2) & (1 << cam1) > 0) and \
       t1_start <= t2_end + cams.dtmax[cam2][cam1] and \
       t2_end + cams.dtmin[cam2][cam1] <= t1_end:
        return True
    return False


def multicam_track_similarity(mtrack1: MulticamTracklet, mtrack2: MulticamTracklet, linkage: str,
                              sims: np.array) -> float:
    """Compute the similarity score between two multicam tracks.

    Parameters
    ----------
    mtrack1: first multi-camera tracklet.
    mtrack2: second multi-camera tracklet.
    linkage: method to use for computing from ('average', 'single', 'complete', 'mean_feature')

    Returns
    -------
    sim_score: similarity between the tracks.
    """

    if linkage == "mean_feature":
        return cosine_sim(mtrack1.mean_feature, mtrack2.mean_feature, True)

    # similarity of all pairs of tracks between mtrack1 and mtrack2
    # this scales badly, but in all sensible cases multicam tracks contain only a few tracks
    all_sims = [sims[t1.idx][t2.idx] for t1 in mtrack1.tracks for t2 in mtrack2.tracks]
    if linkage == "average":
        return np.mean(all_sims)
    if linkage == "single":
        return np.max(all_sims)
    if linkage == "complete":
        return np.min(all_sims)
    raise ValueError("Invalid linkage parameter value.")


def mtmc_clustering(tracks: List[List[Tracklet]],
                    cams: Union[CameraLayout, None],
                    min_sim: float = 0.5,
                    linkage: str = "average") -> List[MulticamTracklet]:
    """Perform multi-camera tracklet clustering.

    Parameters
    ----------
    tracks: single camera tracklets by camera.
    cams: camera layout of the system
    min_sim: minimum similarity score between tracklet mean features to merge (default 0.5)
    linkage: linkage method to use in merging from: ('average', 'single', 'complete')

    Returns
    -------
    mtracks: The final multi-camera tracklets. The single cam tracks contained in each
    are assigned the new ids.
    """

    all_tracks = []
    # add camera info and synchronization info to tracks
    for i, cam_tracks in enumerate(tracks):
        for track in cam_tracks:
            track.cam = i
            track.idx = len(all_tracks)
            if cams:
                track.global_start = track.frames[0] / cams.fps[i] / cams.scales[i] + cams.offset[i]
                track.global_end = track.frames[-1] / cams.fps[i] / cams.scales[i] + cams.offset[i]
            all_tracks.append(track)
    n = len(all_tracks)

    log.info("Starting clustering of %s tracks, precomputing compatibility and similarity matrices ...", n)
    log_start_time = time.time()

    # precompute compatibility between tracks
    compat = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i, n):
            is_comp = tracks_compatible(all_tracks[i], all_tracks[j], cams)
            compat[i, j] = is_comp
            compat[j, i] = is_comp

    def any_compatible(mtrack1: MulticamTracklet, mtrack2: MulticamTracklet) -> bool:
        """Is there a pair of tracks between mtrack1 and mtrack2 that are compatible?"""
        for track1 in mtrack1.tracks:
            for track2 in mtrack2.tracks:
                if compat[track1.idx][track2.idx]:
                    return True
        return False

    # precompute similarities between tracks
    f = torch.Tensor(np.stack([tr.mean_feature for tr in all_tracks]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f.to(device)
    sim = torch.matmul(f, f.T).cpu().numpy()
    log.info(f"Precomputation finished, took {time.time() - log_start_time:.3f} s.")

    # initialize multicam tracklets
    mtracks = [MulticamTracklet(i, [all_tracks[i]], len(tracks)) for i in range(n)]
    remaining_tracks = set(range(n))
    # timestamp of when the last time an mtrack was modified
    last_mod = [0] * n
    timestamp = 1

    # min priority queue for storing possible merges
    # entries are: (-similarity, timestamp, track1_idx, track2_idx)
    merge_queue = []

    # init merge queue
    for i in range(n):
        for j in range(i + 1, n):
            if compat[i][j] and sim[i][j] >= min_sim:
                merge_queue.append((-sim[i][j], timestamp, i, j))

    heapq.heapify(merge_queue)

    while len(merge_queue) > 0:
        minus_sim, t_insert, i1, i2 = heapq.heappop(merge_queue)
        if minus_sim > -min_sim:
            break
        # if since the queue entry any tracks was modified, it is an invalid entry
        if t_insert < max(last_mod[i1], last_mod[i2]):
            continue

        # let's merge the two mtracks
        mtracks[i1].merge_with(mtracks[i2])
        # update their timestamps, and delete the unneeded track
        timestamp += 1
        remaining_tracks.remove(i2)
        last_mod[i1] = timestamp
        last_mod[i2] = timestamp

        # recalculate similarity to other remaining mtracks that can be merged to this one
        for i_other in remaining_tracks:
            if i_other == i1:
                continue
            if have_mutual_cams(mtracks[i1], mtracks[i_other]) or not any_compatible(mtracks[i1], mtracks[i_other]):
                continue
            s = multicam_track_similarity(mtracks[i1], mtracks[i_other], linkage, sim)
            if s >= min_sim:
                heapq.heappush(merge_queue, (-s, timestamp, i1, i_other))

    # drop invalidated mtracks
    mtracks = [mtracks[i] for i in remaining_tracks]

    # reindex final tracks and finalize them
    for i, mtrack in enumerate(mtracks):
        mtrack.id = i
        mtrack.finalize()

    log_total_time = round(time.time() - log_start_time, 3)
    log.info(
        "mtmc clustering took %s seconds: %s final tracks.", log_total_time, len(mtracks))

    return mtracks
