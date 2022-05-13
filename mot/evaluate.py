import numpy as np
import math
import argparse
from pprint import pprint
import motmetrics as mm

from tools.metrics import iou
from tools.conversion import load_motchallenge_format, load_csv_format


def to_frame_list(detection_dict, total_frames=-1):
    if total_frames < 0:
        total_frames = max(detection_dict["frame"]) + 1
    frames = [[] for _ in range(total_frames)]

    for fr, tx, ty, w, h, id_ in zip(detection_dict["frame"],
                                     detection_dict["bbox_topleft_x"],
                                     detection_dict["bbox_topleft_y"],
                                     detection_dict["bbox_width"],
                                     detection_dict["bbox_height"],
                                     detection_dict["track_id"]):
        frames[fr].append((tx, ty, w, h, id_))
    return frames


def greedy_matching(gt_boxes, pred_boxes, min_iou=0.5):
    dists = []
    for i1, b1 in enumerate(gt_boxes):
        for i2, b2 in enumerate(pred_boxes):
            d = iou(b1[:4], b2[:4])
            dists.append((d, i1, i2))
    dists.sort(key=lambda x: x[0], reverse=True)
    matched = []
    gt_matched, pred_matched = set(), set()
    for dist, i1, i2 in dists:
        if dist < min_iou:
            break

        if i1 in gt_matched or i2 in pred_matched:
            continue

        matched.append((i1, i2, dist))
        gt_matched.add(i1)
        pred_matched.add(i2)
    unmatched_gt = set(range(len(gt_boxes))).difference(gt_matched)
    unmatched_pred = set(range(len(pred_boxes))).difference(pred_matched)
    return matched, unmatched_gt, unmatched_pred


def eval_tracking(pred_detections, gt_detections, min_iou=0.5, ignore_fp=False):
    """ Calculate multiple measures to evaluate tracking quality.
    Parameters
    ----------
    frame_detections: dict(list)
        Detection dictionary containing lists for each property of a detection (frame, track_id, etc)
    gt_detections: dict(list)
        Detection dictionary
    Returns
    -------
    dict
        Dictionary containing a key, value for each measure. Possible keys are:
        mota: Mean Object Tracking Accuracy,
        idsw: Number of ID switches,
        motp: Mean Object Tracking Precision,
        mt: Number of Mostly Tracked ids,
        ml: Number of Mostly Lost ids,
        fp: False Positives,
        fn: False Negatives,
        recall: tp / (tp + fn),
        precision: tp / (tp + fp),
        fm: Total Fragmentations,
    """

    total_frames = max(
        max(pred_detections["frame"]), max(gt_detections["frame"])) + 1
    preds_by_frame = to_frame_list(pred_detections, total_frames)
    gt_by_frame = to_frame_list(gt_detections, total_frames)

    idsw = 0
    mota_over = 0
    motp_over = 0
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_fragmentations = 0

    # last predicted id for a ground truth id for tracking id switches
    last_pred_for_id = {}
    lost_gt_track_ids = set()
    gt_track_missed_frames = {}
    gt_track_found_frames = {}

    for gt, preds in zip(gt_by_frame, preds_by_frame):
        matched, unmatched_gt, unmatched_pred = greedy_matching(
            gt, preds, min_iou)
        tp = len(matched)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        idsw_now = 0

        # check for id switch for each matched ground truth box
        for i_gt, i_pred, iou_dist in matched:
            gt_id = gt[i_gt][4]
            pred_id = preds[i_pred][4]
            if gt_id in last_pred_for_id and last_pred_for_id[gt_id] != pred_id:
                idsw_now += 1
            last_pred_for_id[gt_id] = pred_id
            if gt_id in lost_gt_track_ids:
                lost_gt_track_ids.remove(gt_id)
            gt_track_found_frames[gt_id] = gt_track_found_frames.get(
                gt_id, 0) + 1

            motp_over += iou_dist

        # add unmatched ground truth boxes to lost tracks
        for i_gt in unmatched_gt:
            gt_id = gt[i_gt][4]
            if gt_id not in lost_gt_track_ids:
                lost_gt_track_ids.add(gt_id)
                total_fragmentations += 1
            gt_track_missed_frames[gt_id] = gt_track_missed_frames.get(
                gt_id, 0) + 1

        if ignore_fp:
            fp = 0

        mota_over += idsw_now + fp + fn
        idsw += idsw_now
        total_gt += len(gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    mostly_tracked = 0
    mostly_lost = 0
    all_gt_ids = set(gt_track_missed_frames).union(set(gt_track_found_frames))
    for gtid in all_gt_ids:
        tracked_ratio = gt_track_found_frames.get(gtid, 0) / (gt_track_found_frames.get(gtid, 0) +
                                                              gt_track_missed_frames.get(gtid, 0))
        if tracked_ratio >= 0.8:
            mostly_tracked += 1
        elif tracked_ratio <= 0.2:
            mostly_lost += 1

    return {
        "mota": 100 * (1 - mota_over / total_gt),
        "motp": motp_over / total_tp if total_tp > 0 else 0.0,
        "idsw": idsw,
        "mt": mostly_tracked,
        "ml": mostly_lost,
        "recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
        "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        "fm": total_fragmentations,
        "fp": total_fp,
        "fn": total_fn,
        "tp": total_tp,
    }


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate tracking against a ground truth file in MOTChallenge format")
    parser.add_argument("--gt", required=True,
                        help="ground truth file in MOTChallenge csv format")
    parser.add_argument("--pred", required=True,
                        help="tracking results in csv")
    parser.add_argument("--min_iou", type=float, default=0.5,
                        help="minimal IOU for assigning a detection to a ground truth box")
    parser.add_argument("--ignore_fp", action="store_true",
                        help="""Set false positives to zero in all frames. This is needed if the ground truth
                        does not include all possible objects (cars), e.g in the CityFlow dataset""")
    parser.add_argument("--print_metric_names", action="store_true")
    parser.add_argument("--own_solver", action="store_true",
                        help="use implementation in this script instead of the motmetrics package, this supports the ignore_fp parameter")
    args = parser.parse_args()

    gt_dict = load_motchallenge_format(args.gt)
    pred_dict = load_csv_format(args.pred)
    if args.own_solver:
        result = eval_tracking(
            pred_dict, gt_dict, args.min_iou, args.ignore_fp)
        pprint(result)
    else:
        result = evaluate_mm(pred_dict, gt_dict, args.min_iou)
        print(result)

    if args.print_metric_names:
        metric_names = """
        IDF1: IDF1 score
        IDP: ID Precision
        IDR: ID Recall
        Rcll: Recall
        Prcn: Precision
        GT: num_unique (number of ground truth tracks)
        MT: Mostly tracked (found in >=80%)
        PT: Partially tracked (found in <80%, >=20%)
        ML: Mostly lost (found in <20%)
        FP: False positives
        FN: False negatives
        IDs: ID switches
        FM: Fragmentations
        MOTA: Mean Object Tracking Accuracy
        MOTP: Mean Object Tracking Precision
        IDt: num_transfer
        IDa: num_acend
        IDm: num_migrate
        """
        print(metric_names)
