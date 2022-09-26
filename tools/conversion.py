from typing import Union
import pandas as pd
from mot.tracklet import Tracklet


def to_frame_list(detections: Union[pd.DataFrame, dict], total_frames=-1):
    """Convert a dict or df describing detections to a list containing info frame-by-frame."""
    if total_frames < 0:
        total_frames = max(detections["frame"]) + 1
    frames = [[] for _ in range(total_frames)]

    for fr, tx, ty, w, h, id_ in zip(detections["frame"],
                                     detections["bbox_topleft_x"],
                                     detections["bbox_topleft_y"],
                                     detections["bbox_width"],
                                     detections["bbox_height"],
                                     detections["track_id"]):
        frames[fr].append((tx, ty, w, h, id_))
    return frames


def detection_dict_to_list(det_dict):
    """Converts frame-by-frame detections from dict of columns to list of rows format."""
    if "conf" in det_dict:
        conf = det_dict["conf"]
    else:
        conf = [1] * len(det_dict["frame"])

    res = []
    for fr, trid, tl_x, tl_y, w, h, c in zip(det_dict["frame"],
                                             det_dict["track_id"],
                                             det_dict["bbox_topleft_x"],
                                             det_dict["bbox_topleft_y"],
                                             det_dict["bbox_width"],
                                             det_dict["bbox_height"],
                                             conf):
        res.append((fr, trid, tl_x, tl_y, w, h, c))
    return res


def detection_list_to_dict(det_list):
    keys = ["frame", "track_id", "bbox_topleft_x", "bbox_topleft_y", "bbox_width",
            "bbox_height", "conf"]
    res = {k: [] for k in keys}
    for det in det_list:
        res["frame"].append(det[0])
        res["track_id"].append(det[1])
        res["bbox_topleft_x"].append(det[2])
        res["bbox_topleft_y"].append(det[3])
        res["bbox_width"].append(det[4])
        res["bbox_height"].append(det[5])
        res["conf"].append(det[6])
    return res


def detection_dict_to_tracklets(det_dict):
    tracklet_dict = {}
    frame = det_dict["frame"]
    tlx = det_dict["bbox_topleft_x"]
    tly = det_dict["bbox_topleft_y"]
    w = det_dict["bbox_width"]
    h = det_dict["bbox_height"]
    track_id = det_dict["track_id"]
    zone = [] if "zone" not in det_dict else det_dict["zone"]
    conf = [] if "conf" not in det_dict else det_dict["conf"]

    for i in range(len(det_dict["frame"])):
        tracklet = tracklet_dict.setdefault(track_id[i], Tracklet(track_id[i]))
        tracklet.frames.append(frame[i])
        tracklet.bboxes.append([tlx[i], tly[i], w[i], h[i]])
        if zone:
            tracklet.zones.append(zone[i])
        if conf:
            tracklet.conf.append(conf[i])
    return list(tracklet_dict.values())


def load_motchallenge_format(file_path, frame_offset=1):
    """Loads a MOTChallenge annotation txt, with frame_offset being the index of the first frame of the video"""
    res = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")
            line = [int(x) for x in line[:6]] + [float(x) for x in line[6:]]

            # Subtract frame offset from frame indices. If indexing starts at one, we convert
            # it to start from zero.
            line[0] -= frame_offset
            res.append(tuple(line))
    return detection_list_to_dict(res)


def load_csv_format(file_path):
    df = pd.read_csv(file_path)
    res = {c: list(df[c]) for c in df.columns}
    return res


def csv_files_to_cityflow(file_paths, cam_idxes, out_path=None):
    dfs = []
    for path, cam in zip(file_paths, cam_idxes):
        df = pd.read_csv(path)
        df["camera"] = cam
        df = df[["camera", "track_id", "frame", "bbox_topleft_x",
                 "bbox_topleft_y", "bbox_width", "bbox_height"]]
        dfs.append(df)
    df = pd.concat(dfs)
    df["frame"] = list(map(lambda x: x + 1, df["frame"]))
    df["xw"] = -1
    df["yw"] = -1
    if out_path:
        df.to_csv(out_path, index=False, header=False, sep=" ")
    return df
