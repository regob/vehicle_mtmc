from typing import List
import numpy as np
import pandas as pd
import motmetrics as mm

from tools.metrics import iou
from tools.conversion import to_frame_list, load_motchallenge_format
from evaluate.experimental import greedy_matching


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates of the same id in the same frame (should never happen)"""
    return df.drop_duplicates(subset=["frame", "track_id"], keep="first")

def remove_single_cam_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Remove tracks from df that only appear on one camera ('cam' column needed)"""
    subdf = df[["cam", "track_id"]].drop_duplicates()
    track_cnt = subdf[["track_id"]].groupby(["track_id"]).size()
    good_ids = track_cnt[track_cnt > 1].index
    return df[df["track_id"].isin(good_ids)]


def load_annots(paths: List[str]) -> pd.DataFrame:
    """Load one txt annot for each camera, and return them in a merged dataframe."""
    dicts = [load_motchallenge_format(path) for path in paths]
    dfs = [pd.DataFrame(d) for d in dicts]
    max_frame = 0
    for i, df in enumerate(dfs):
        df["frame"] = df["frame"].apply(lambda x: x + max_frame)
        df["cam"] = i
        max_frame = max(df["frame"])
    df = pd.concat(dfs)
    return remove_duplicates(df)


def evaluate_dfs(test_df: pd.DataFrame, pred_df: pd.DataFrame, min_iou=0.5, ignore_fp=False):
    """Evaluate MOT (or merged MTMC) predictions against the ground truth annotations."""

    acc = mm.MOTAccumulator(auto_id=True)

    total_frames = max(max(pred_df["frame"]), max(test_df["frame"])) + 1
    test_by_frame = to_frame_list(test_df, total_frames)
    pred_by_frame = to_frame_list(pred_df, total_frames)

    for gt, preds in zip(test_by_frame, pred_by_frame):
        mat_gt = np.array([x[:4] for x in gt])
        mat_pred = np.array([x[:4] for x in preds])
        iou_matrix = mm.distances.iou_matrix(mat_gt, mat_pred, 1 - min_iou)
        n, m = len(gt), len(preds)

        if ignore_fp:
            # remove preds that are unmatched (would be false positives)
            matched_gt, matched_pred = mm.lap.linear_sum_assignment(iou_matrix)
            remain_preds = set(matched_pred)
            remain_pred_idx = [-1] * m
            for i, p in enumerate(remain_preds):
                remain_pred_idx[p] = i
            m = len(remain_preds)

            # now we can create the distance matrix rigged for our matching
            iou_matrix = np.full((n, m), np.nan)
            for i_gt, i_pred in zip(matched_gt, matched_pred):
                iou_matrix[i_gt, remain_pred_idx[i_pred]] = 0.0
        else:
            remain_pred_idx = list(range(m))

        pred_ids = [x[4]
                    for i, x in enumerate(preds) if remain_pred_idx[i] >= 0]
        gt_ids = [x[4] for x in gt]
        acc.update(gt_ids, pred_ids, iou_matrix)

    # acc = mm.utils.compare_to_groundtruth(
    #     test_df, pred_df, "iou", distth=(1 - min_iou))

    metrics = mm.metrics.motchallenge_metrics
    metrics.extend(["idfp", "idfn", "idtp"])
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=metrics, name="MTMC")
    return summary


def formatted_summary(summary):
    mh = mm.metrics.create()
    formatters = mh.formatters
    formatters["motp"] = lambda motp: "{:.2%}".format(1 - motp)
    strsummary = mm.io.render_summary(summary, formatters=formatters,
                                      namemap=mm.io.motchallenge_metric_names)
    return strsummary


def evaluate_mm(pred_detections, gt_detections, min_iou=0.5):
    total_frames = max(
        max(pred_detections["frame"]), max(gt_detections["frame"])) + 1
    preds_by_frame = to_frame_list(pred_detections, total_frames)
    gt_by_frame = to_frame_list(gt_detections, total_frames)

    acc = mm.MOTAccumulator(auto_id=True)
    for gt, preds in zip(gt_by_frame, preds_by_frame):
        dists = []
        for gt_det in gt:
            dists.append([])
            for pred_det in preds:
                sim = iou(gt_det[:4], pred_det[:4])
                if sim < min_iou:
                    dists[-1].append(np.nan)
                else:
                    dists[-1].append(1 - sim)
        gt_ids = [gt_det[4] for gt_det in gt]
        pred_ids = [pred_det[4] for pred_det in preds]

        acc.update(gt_ids, pred_ids, dists)

    mh = mm.metrics.create()
    summary_1 = mh.compute(
        acc, metrics=mm.metrics.motchallenge_metrics, name="mot")

    formatters = mh.formatters
    formatters["motp"] = lambda motp: "{:.2%}".format(1 - motp)

    strsummary = mm.io.render_summary(summary_1, formatters=formatters,
                                      namemap=mm.io.motchallenge_metric_names)
    return strsummary
