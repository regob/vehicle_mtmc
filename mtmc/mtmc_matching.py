from typing import List
import time


from mot.tracklet import Tracklet
from mtmc.cameras import CameraLayout
from mtmc.multicam_tracklet import MulticamTracklet
from tools.metrics import cosine_sim
from tools.data_structures import DSU
from tools import log


def _flatten_tracks_with_cam_info(tracks: List[List[Tracklet]], cams: CameraLayout) -> List[Tracklet]:
    """Save camera info and global timestamps in tracklets, and flatten them."""
    flat_tracks = []
    for i, cam_tracks in enumerate(tracks):
        for track in cam_tracks:
            track.cam = i
            track.global_start = track.frames[0] / cams.fps[i] + cams.offset[i]
            track.global_end = track.frames[-1] / cams.fps[i] + cams.offset[i]
            flat_tracks.append(track)
    return flat_tracks


def greedy_mtmc_matching(tracks: List[List[Tracklet]], cams: CameraLayout, min_sim=0.5) -> List[MulticamTracklet]:
    """Run greedy matching on single-camera tracks.

    Args:
       tracks: single camera tracklets by camera.
       cams: camera layout of the system
       min_sim: minimum similarity score between tracklet mean features to merge (default 0.5)

    Returns:
       List[MulticamTracklet]: the final multi-camera tracklets.
    """
    flat_tracks = _flatten_tracks_with_cam_info(tracks, cams)
    flat_tracks.sort(key=lambda x: x.global_end)

    log.debug("Starting greedy mtmc matching of %s tracks.",
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

    # iterate tracks increasing by their global_end timestamp
    for track in flat_tracks:

        log.debug("Checking track: %s from cam %s", track, track.cam)

        mtrack_idx = track_dsu.find_root(track.dsu_idx)
        mtrack = multicam_tracks[mtrack_idx]
        cams_possible_bmp = cams.cam_compatibility_bitmap(
            track.cam) & mtrack.inverse_cams()

        # iterate all cameras to check matching tracklets
        for cam in range(cams.n_cams):
            if not ((1 << cam) & cams_possible_bmp):
                continue
            min_start = track.global_end + cams.dtmin[track.cam][cam]
            max_start = track.global_end + cams.dtmax[track.cam][cam]

            # we check candidates that start between the given timestamps
            candidates = _get_tracks_start_between(
                tracks[cam], min_start, max_start)
            candidates_w_scores = []
            for cand in candidates:
                cand_mtrack_idx = track_dsu.find_root(cand.dsu_idx)
                cand_mtrack = multicam_tracks[cand_mtrack_idx]

                # if the multicam tracklets have a common camera, we cannot merge them
                if cand_mtrack.cams & mtrack.cams:
                    continue

                # add the candidate with its similarity score to the list of final candidates
                candidates_w_scores.append((cand_mtrack_idx,
                                            cosine_sim(cand_mtrack.mean_feature, mtrack.mean_feature)))
            log.debug("Candidates: %s", candidates_w_scores)
            if len(candidates_w_scores) == 0:
                continue
            best_cand_idx, best_sim = max(
                candidates_w_scores, key=lambda x: x[1])

            # merge with the candidate
            if best_sim >= min_sim:
                track_dsu.union_sets(mtrack_idx, best_cand_idx)
                new_root = track_dsu.find_root(mtrack_idx)
                if new_root == mtrack_idx:
                    mtrack.merge_with(multicam_tracks[best_cand_idx])
                else:
                    mtrack, old_mtrack = multicam_tracks[best_cand_idx], mtrack
                    mtrack.merge_with(old_mtrack)
                    mtrack_idx = new_root
                    cams_possible_bmp = cams.cam_compatibility_bitmap(
                        track.cam) & mtrack.inverse_cams()

    # filter multicam tracks to keep only those that are valid (= it is a root of a set)
    valid_mtracks = [multicam_tracks[idx] for idx in range(
        len(multicam_tracks)) if track_dsu.find_root(idx) == idx]

    # reindex final tracks
    for i, mtrack in enumerate(valid_mtracks):
        mtrack.id = i

    log_total_time = round(time.time() - log_start_time, 3)
    log.debug(
        "greedy mtmc matching took %s seconds: %s final tracks.", log_total_time, len(valid_mtracks))

    return valid_mtracks


def _get_tracks_start_between(tracks: List[Tracklet], min_start: int, max_start: int) -> List[Tracklet]:
    """Fetch tracks that start between given timestamps from a sorted list of tracks."""
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


if __name__ == "__main__":
    cams = CameraLayout("../config/zala/mtmc_camera_layout.txt")
    import pickle

    def load(pth):
        f = open(pth, "rb")
        res = pickle.load(f)
        f.close()
        return res
    cam_tracks = [
        load("../output/balaton/zala_balaton_results.pkl"),
        load("../output/gasparich/gasparich_results.pkl"),
        load("../output/kormend/kormend_results.pkl"),
        load("../output/bevasarlokozpont/bevasarlokozpont_results.pkl"),
    ]
    for tracks in cam_tracks:
        for track in tracks:
            track.compute_mean_feature()
    print(f"Total tracks: {sum(len(tr) for tr in cam_tracks)}")

    res = greedy_mtmc_matching(cam_tracks, cams)
    print(len(res))
