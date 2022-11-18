from typing import List, Set
import time
import heapq
import numpy as np


from mot.tracklet import Tracklet
from mtmc.cameras import CameraLayout
from mtmc.multicam_tracklet import MulticamTracklet, have_mutual_cams
from tools.metrics import cosine_sim
from tools.data_structures import DSU
from tools import log


def _flatten_tracks_with_cam_info(tracks: List[List[Tracklet]], cams: CameraLayout) -> List[Tracklet]:
    """Save camera info and global timestamps in tracklets, and flatten them."""
    flat_tracks = []
    for i, cam_tracks in enumerate(tracks):
        for track in cam_tracks:
            track.cam = i
            track.global_start = track.frames[0] / cams.fps[i] / cams.scales[i] + cams.offset[i]
            track.global_end = track.frames[-1] / cams.fps[i] / cams.scales[i] + cams.offset[i]
            flat_tracks.append(track)
    return flat_tracks


def multicam_track_similarity(mtrack1: MulticamTracklet, mtrack2: MulticamTracklet, linkage: str) -> float:
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
    all_sims = [cosine_sim(t1.mean_feature, t2.mean_feature, True) for t1 in mtrack1.tracks for t2 in mtrack2.tracks]
    if linkage == "average":
        return np.mean(all_sims)
    if linkage == "single":
        return np.max(all_sims)
    if linkage == "complete":
        return np.min(all_sims)
    raise ValueError("Invalid linkage parameter value.")

        
def greedy_mtmc_matching(tracks: List[List[Tracklet]], cams: CameraLayout, min_sim=0.5, linkage="average") -> List[MulticamTracklet]:
    """Run greedy matching on single-camera tracks.

    Run a similar merging algorithm to agglomerative clustering with a distance limit to merge tracks.
    The camera constraints are also respected, thus two sets of tracks (multicam tracks) can be matched if:
        * they do not share any cameras
        * we can choose a (track1, track2) pair, so that track1 and track2 are from different multicam tracks and
          track1.global_end + dtmin <= track2.global_start <= track1.global_end + dtmax,
          where dtmin and dtmax are the specific values for track1.cam and track2.cam,
          also track1.cam and track2.cam are compatible in this order.
        * their similarity is higher than min_sim, with respect to 'linkage' in calculation.
    First each single camera track has a new multicam track, then we continue merging the pair,
    with the highest similarity until it is lower than min_sim.
    
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
    flat_tracks = _flatten_tracks_with_cam_info(tracks, cams)
    # flat_tracks.sort(key=lambda x: x.global_end)

    log.info("Starting greedy mtmc matching of %s tracks.",
              len(flat_tracks))
    log_start_time = time.time()

    # sort tracks for each camera by start timestamps
    for track_list in tracks:
        track_list.sort(key=lambda x: x.global_start)

    # We initialize a multicam tracklet for each track, then start merging them.
    # Some indices will be unused in multicam_tracks as we start merging, but the
    # DSU shows the correct index for each track
    track_dsu = DSU(len(flat_tracks))
    multicam_tracks = [MulticamTracklet(
        i, [track], cams.n_cams) for i, track in enumerate(flat_tracks)]
    for i, track in enumerate(flat_tracks):
        track.dsu_idx = i

    # min priority queue for storing possible merges
    # entries are: (-similarity, timestamp, track1, track2)
    merge_queue = []

    # maintain a timestamp to check whether a tuple retrieved from the queue is valid
    # (if either track1 or track2 was modified since insertion, it is invalid)
    for tr in multicam_tracks:
        tr.timestamp = 0
    timestamp = 1

    # initialize the queue with all valid pairs
    for track in flat_tracks:
        mtrack = multicam_tracks[track.dsu_idx]
        candidates = _get_track_candidates(mtrack, tracks, cams)
        for cand in candidates:
            cand_mtrack = multicam_tracks[cand.dsu_idx]
            merge_queue.append((-multicam_track_similarity(mtrack, cand_mtrack, linkage), timestamp, track.dsu_idx, cand.dsu_idx))
        
    # initialize the heap from the list
    heapq.heapify(merge_queue)

    # try to merge while the similarity is over min_sim
    while len(merge_queue) > 0:
        minus_sim, entry_timestamp, mtrack1_idx, mtrack2_idx = heapq.heappop(merge_queue)
        if minus_sim >= -min_sim:
            break

        mtrack1 = multicam_tracks[mtrack1_idx]
        mtrack2 = multicam_tracks[mtrack2_idx]
        if entry_timestamp < mtrack1.timestamp or entry_timestamp < mtrack2.timestamp:
            continue
        if have_mutual_cams(mtrack1, mtrack2):
            continue
        
        # merge mtrack1 and mtrack2
        timestamp += 1
        mtrack1.timestamp = timestamp
        mtrack2.timestamp = timestamp
        track_dsu.union_sets(mtrack1_idx, mtrack2_idx)
        new_root = track_dsu.find_root(mtrack1_idx)
        if new_root == mtrack2_idx:
            mtrack1_idx, mtrack1, mtrack2_idx, mtrack2 = mtrack2_idx, mtrack2, mtrack1_idx, mtrack1
        mtrack1.merge_with(mtrack2)

        log.debug("Merged tracks (sim=%s): %s", -minus_sim, list(map(lambda x: (x.cam, x.track_id),mtrack1.tracks)))

        # insert new merge entries
        timestamp += 1
        cands =  _get_track_candidates(mtrack1, tracks, cams)
        cand_mtracks = set(track_dsu.find_root(c.dsu_idx) for c in cands)
        for cand_idx in cand_mtracks:
            mtrack3 = multicam_tracks[cand_idx]
            if have_mutual_cams(mtrack1, mtrack3):
                continue
            
            heapq.heappush(merge_queue, (-multicam_track_similarity(mtrack1, mtrack3, linkage), timestamp, mtrack1_idx, cand_idx))


    # filter multicam tracks to keep only those that are valid (= it is a root of a set)
    valid_mtracks = [multicam_tracks[idx] for idx in range(
        len(multicam_tracks)) if track_dsu.find_root(idx) == idx]

    # reindex final tracks and finalize them
    for i, mtrack in enumerate(valid_mtracks):
        mtrack.id = i
        mtrack.finalize()

    log_total_time = round(time.time() - log_start_time, 3)
    log.info(
        "greedy mtmc matching took %s seconds: %s final tracks.", log_total_time, len(valid_mtracks))

    return valid_mtracks
    
    
def _get_track_candidates(current_track: MulticamTracklet, tracks: List[List[Tracklet]], cams: CameraLayout) -> Set[Tracklet]:
    """Return the candidates for matching with the current_track from other cameras."""
    excluded_cams = set(t.cam for t in current_track.tracks)
    candidates = set()

    for track in current_track.tracks:
        compat = cams.cam_compatibility_bitmap(track.cam)
        for cam in range(cams.n_cams):
            if cam in excluded_cams or (compat & (1 << cam)) == 0:
                continue
            dtmin = cams.dtmin[track.cam][cam]
            dtmax = cams.dtmax[track.cam][cam]
            min_start, max_start = track.global_end + dtmin, track.global_end + dtmax

            for tr in _get_tracks_start_between(tracks[cam], min_start, max_start):
                candidates.add(tr)

    return candidates

def _get_tracks_start_between(tracks: List[Tracklet], min_start: int, max_start: int) -> List[Tracklet]:
    """Fetch tracks that start between given timestamps from a sorted list of tracks."""
    if len(tracks) == 0:
        return []
    i, j = 0, len(tracks) - 1
    while j - i > 1:
        mid = (i + j) >> 1
        if tracks[mid].global_start < min_start:
            i = mid + 1
        else:
            j = mid
    if tracks[i].global_start >= min_start:
        start = i
    elif tracks[j].global_start >= min_start:
        start = j
    else:
        return []

    i, j = start, len(tracks) - 1
    while j - i > 1:
        mid = (i + j) >> 1
        if tracks[mid].global_start <= max_start:
            i = mid + 1
        else:
            j = mid
    if tracks[i].global_start > max_start:
        end = i
    elif tracks[j].global_start > max_start:
        end = j
    else:
        end = j + 1

    return tracks[start:end]
