import numpy as np
import pandas as pd
import pickle
from bisect import bisect_left
from mot.tracklet import Tracklet
from tools.metrics import iou


def save_tracklets(tracklets, path, max_features=None):
    """Saves tracklets using pickle (with re-id features)"""
    if max_features is not None:
        for tracklet in tracklets:
            tracklet.cluster_features(max_features)
    with open(path, "wb") as fp:
        pickle.dump(tracklets, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_tracklets(pickled_path):
    """Loads a pickled list of tracklets."""
    with open(pickled_path, "rb") as f:
        tracklets = pickle.load(f)
    return tracklets


def to_detections(tracklets):
    res = {
        "frame": [],
        "bbox_topleft_x": [],
        "bbox_topleft_y": [],
        "bbox_width": [],
        "bbox_height": [],
        "track_id": [],
    }
    if len(tracklets) == 0:
        return res
    
    for k in tracklets[0].static_attributes:
        res[k] = []
    for k in tracklets[0].dynamic_attributes:
        res[k] = []
    if tracklets[0].zones:
        res["zone"] = []

    for tracklet in tracklets:
        res["frame"].extend(tracklet.frames)
        for x, y, w, h in tracklet.bboxes:
            res["bbox_topleft_x"].append(int(x))
            res["bbox_topleft_y"].append(int(y))
            res["bbox_width"].append(int(round(w)))
            res["bbox_height"].append(int(round(h)))
        res["track_id"].extend([tracklet.track_id] * len(tracklet.frames))
        for static_f, val in tracklet.static_attributes.items():
            values = val if isinstance(val, list) else [
                val] * len(tracklet.frames)
            res[static_f].extend(values)
        for dynamic_f, val in tracklet.dynamic_attributes.items():
            res[dynamic_f].extend(val)
        if tracklet.zones:
            res["zone"].extend(tracklet.zones)

    # all columns should have the same length
    lengths = list(map(len, res.values()))
    lengths_equal = list(map(lambda l: l == lengths[0], lengths))
    if not all(lengths_equal):
        for k, v in res.items():
            print(f"Items in column {k}: {len(v)}")
        raise ValueError("Error: not all column lengths are equal.")

    return res


def save_tracklets_csv(tracklets, path):
    res = to_detections(tracklets)
    df = pd.DataFrame(res)
    df.to_csv(path, index=False)


def split_tracklet(tracklet: Tracklet, frame_idx: int, new_track_id: int) -> Tracklet:
    """ Split a tracklet into two parts at a given frame index.
    Parameters
    ----------
    tracklet: Tracklet
    frame_idx: int
        The index of the first frame in the tracklet, which belongs to the
        second track after splitting.
    min_track_id: int
        Track_id for the new tracks. The second one will get min_track_id+1
    Returns
    -------
    tuple
        A pair of new Tracklet objects.
    """

    track1 = Tracklet(new_track_id)
    track2 = Tracklet(new_track_id + 1)

    track1.features = tracklet.features[:frame_idx]
    track1.frames = tracklet.frames[:frame_idx]
    track1.bboxes = tracklet.bboxes[:frame_idx]
    track1.zones = tracklet.zones[:frame_idx]
    track1.conf = tracklet.conf[:frame_idx]
    track1.static_attributes = {k: v if isinstance(v, int) else v[:frame_idx]
                                for k, v in tracklet.static_attributes.items()}
    track1.dynamic_attributes = {k: v[:frame_idx]
                                 for k, v in tracklet.dynamic_attributes.items()}

    track2.features = tracklet.features[frame_idx:]
    track2.frames = tracklet.frames[frame_idx:]
    track2.bboxes = tracklet.bboxes[frame_idx:]
    track2.zones = tracklet.zones[frame_idx:]
    track2.conf = tracklet.conf[frame_idx:]
    track2.static_attributes = {k: v if isinstance(v, int) else v[frame_idx:]
                                for k, v in tracklet.static_attributes.items()}
    track2.dynamic_attributes = {k: v[frame_idx:]
                                 for k, v in tracklet.dynamic_attributes.items()}
    return track1, track2


def join_tracklets(track1, track2):
    """ Merges two tracklets. The second is appended to the first one, and the first track's id is kept. """
    track1.frames.extend(track2.frames)
    track1.features.extend(track2.features)
    track1.bboxes.extend(track2.bboxes)
    track1.zones.extend(track2.zones)
    for feature in track1.static_attributes:
        attr = track1.static_attributes[feature]
        if isinstance(attr, int):
            continue
        attr.extend(track2.static_attributes[feature])
    for feature in track1.dynamic_attributes:
        track1.dynamic_attributes[feature].extend(
            track2.dynamic_attributes[feature])
    return track1


def refine_tracklets(tracklets, zone_matcher, verbose=True):
    """ Split and join tracklets based on zone and frame criterions, and return all valid and invalid tracklets. """

    if verbose:
        print("Running refinement... Initial tracklets: {}".format(len(tracklets)))

    MIN_SPLIT_DIST = 0.25
    MAX_MERGE_DIST = 0.13
    MAX_FRAME_GAP = 5
    MIN_IOU_MERGE = 0.2

    valid, invalid = [], []
    for tracklet in tracklets:
        if zone_matcher.is_valid_path(tracklet.zones):
            valid.append(tracklet)
        else:
            invalid.append(tracklet)

    if verbose:
        print("Initial valid: {}, invalid: {}".format(len(valid), len(invalid)))

    next_track_idx = max(map(lambda t: t.track_id, tracklets)) + 1
    invalid_new = []
    for tracklet in invalid:
        last_frame = tracklet.frames[0]
        split_idx, valid_num, valid_len = -1, 0, 0
        for idx, frame in enumerate(tracklet.frames[1:]):
            if frame - last_frame > 1:
                zones1, zones2 = tracklet.zones[:idx +
                                                1], tracklet.zones[idx + 1:]
                valid1 = zone_matcher.is_valid_path(zones1)
                valid2 = zone_matcher.is_valid_path(zones2)
                if (valid1 + valid2 > valid_num) or \
                   (valid1 + valid2 == valid_num and
                        ((valid1 and len(zones1) > valid_len) or (valid2 and len(zones2) > valid_len))):
                    split_idx = idx + 1
                    valid_num = valid1 + valid2
                    if valid1 and valid2:
                        valid_len = max(len(zones1), len(zones2))
                    elif valid1:
                        valid_len = len(zones1)
                    else:
                        valid_len = len(zones2)
            last_frame = frame

        if valid_num > 0:
            track1, track2 = split_tracklet(
                tracklet, split_idx, next_track_idx)
            next_track_idx += 2
            for track in [track1, track2]:
                if zone_matcher.is_valid_path(track.zones):
                    valid.append(track)
                else:
                    invalid_new.append(track)
        else:
            invalid_new.append(tracklet)

    if verbose:
        print("First round - valid: {}, invalid: {}".format(len(valid), len(invalid_new)))

    tracklet_chunks = []
    for tracklet in invalid_new:
        split_idxes = []
        last_frame, steps = tracklet.frames[0], 0
        mean_feature = tracklet.features[0]
        for idx, (feature, frame) in enumerate(zip(tracklet.features, tracklet.frames)):
            dist = 1 - np.dot(mean_feature, feature)

            # insert a split point at this index
            if steps >= 5 and dist > MIN_SPLIT_DIST and frame - last_frame > 1:
                split_idxes.append(idx)
                steps = 0

            last_frame = frame
            steps += 1
            mean_feature = 0.7 * mean_feature + 0.3 * feature
            mean_feature = mean_feature / np.linalg.norm(mean_feature, 2)

        prev_idx = 0
        rem_track = tracklet
        for idx in split_idxes:
            if verbose:
                print(f"Splitting track {tracklet.track_id} at {idx}.")

            track1, rem_track = split_tracklet(
                rem_track, idx - prev_idx, next_track_idx)
            next_track_idx += 2
            tracklet_chunks.append(track1)
            prev_idx = idx
        tracklet_chunks.append(rem_track)

    # sort tracklets into chronological order by last frames
    tracklet_chunks.sort(key=lambda track: track.frames[-1])
    tracklet_last_frames = list(map(lambda tr: tr.frames[-1], tracklet_chunks))
    tracklet_taken = [False] * len(tracklet_chunks)
    final_invalid = []

    if verbose:
        print("Total tracklet chunks: {}".format(len(tracklet_chunks)))

    # join tracklet chunks if they seem to follow each other
    for idx in range(len(tracklet_chunks)):
        if tracklet_taken[idx]:
            continue
        tracklet_taken[idx] = True
        tracklet = tracklet_chunks[idx]
        candidate = bisect_left(
            tracklet_last_frames, tracklet.frames[-1] + 1)

        while candidate < len(tracklet_chunks) and \
                tracklet_last_frames[candidate] - 1 - tracklet.frames[-1] <= MAX_FRAME_GAP:
            if tracklet_taken[candidate]:
                candidate += 1

            cand_tr = tracklet_chunks[candidate]

            # merge tracklets and check whether it is valid
            if iou(tracklet.bboxes[-1], cand_tr.bboxes[0]) >= MIN_IOU_MERGE and \
               (1 - np.dot(tracklet.features[-1], cand_tr.features[-1])) <= MAX_MERGE_DIST:
                tracklet_taken[candidate] = True
                tracklet = join_tracklets(tracklet, cand_tr)
                if zone_matcher.is_valid_path(tracklet.zones):
                    break
                candidate = bisect_left(
                    tracklet_last_frames, tracklet.frames[-1] + 1)
            else:
                candidate += 1

        if zone_matcher.is_valid_path(tracklet.zones):
            valid.append(tracklet)
        else:
            final_invalid.append(tracklet)

    if verbose:
        print("Processing done. Valid: {}, invalid: {}".format(
            len(valid), len(final_invalid)))

    return (valid, final_invalid)
