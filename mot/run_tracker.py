import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import imageio
import argparse
import matplotlib.pyplot as plt
import time

try:
    import cv2
except ImportError as e:
    print("cv2 import failed, on-the-fly video output not available: {}".format(e))

# repository imports (PYTHONPATH needs to be set)
from mot.deep_sort import preprocessing, nn_matching
from mot.deep_sort.detection import Detection
from mot.deep_sort.tracker import Tracker
from mot.tracklet import Tracklet
from mot.tracklet_processing import save_tracklets, save_tracklets_csv
from mot.static_features import StaticFeatureExtractor, FEATURES

from reid.feature_extractor import FeatureExtractor
from reid.vehicle_reid.load_model import load_model_from_opts

from detection.load_detector import load_yolo

from tools.util import FrameRateCounter, Timer
from tools.preprocessing import create_extractor
from config.defaults import get_cfg_defaults


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Multi-object tracker on a video.")
    parser.add_argument("--config", help="config yaml file")
    return parser.parse_args()


def draw_rectangle(img_np, tx, ty, w, h, color, width):
    """ Draw a colored rectangle to a numpy image. The top left corner is at (x,y), the rect has
    a width of w and a height of h. 'width' is the border width of the rectangle. """

    color = np.array(color)

    # bottom right coordinates
    bx, by = int(tx + w), int(ty + h)

    ranges = np.array([[int(ty - width), ty, int(tx - width), int(bx + width + 1)],
                       [by + 1, int(by + width + 1),
                        int(tx - width), int(bx + width + 1)],
                       [int(ty - width), int(by + width + 1),
                        int(tx - width), tx],
                       [int(ty - width), int(by + width + 1), bx + 1, int(bx + width + 1)]])

    ranges[np.where(ranges < 0)] = 0

    for r in ranges:
        img_np[r[0]:r[1], r[2]:r[3]] = color
    return img_np


def put_text(img_pil, text, x, y, color, font):
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, (color[0], color[1], color[2],
                             255), font=font)
    return img_pil


def annotate(img_pil, id_label, static_features, x, y, color, font):
    draw = ImageDraw.Draw(img_pil)
    text = [id_label] + [f"{k}: {FEATURES[k][v]}" for k,
                         v in static_features.items()]
    text = "\n".join(text)
    draw.multiline_text(
        (x, y), text, (color[0], color[1], color[2], 255), font=font)
    return img_pil


def filter_boxes(boxes, scores, classes, good_classes, min_confid=0.5, mask=None):
    good_boxes = []
    for bbox, score, cl in zip(boxes, scores, classes):
        if score < min_confid or cl not in good_classes:
            continue
        good_boxes.append(bbox)

    if mask is None:
        return good_boxes

    final_boxes = []
    for bbox in good_boxes:
        cx, cy = int(bbox[0]), int(bbox[1])
        if mask[cy, cx, 0] > 0:
            final_boxes.append(bbox)
    return final_boxes


########################################
# Parse args and configuration
########################################

args = parse_args()
cfg = get_cfg_defaults()
if args.config:
    cfg.merge_from_file(os.path.join(cfg.SYSTEM.CFG_DIR, args.config))
cfg.freeze()


########################################
# Loading models, initialization
########################################

max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 0.85

device = torch.device(
    "cpu") if cfg.SYSTEM.NUM_GPUS == 0 or not torch.cuda.is_available() else torch.device("cuda")

# initialize reid model
reid_model = load_model_from_opts(os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.MOT.REID_MODEL_OPTS),
                                  ckpt=os.path.join(
                                      cfg.SYSTEM.ROOT_DIR, cfg.MOT.REID_MODEL_CKPT),
                                  remove_classifier=True)
if not cfg.MOT.REID_FP16:
    reid_model.half()
reid_model.to(device)
reid_model.eval()
extractor = create_extractor(FeatureExtractor, batch_size=cfg.MOT.REID_BATCHSIZE,
                             model=reid_model)


# initialize deep_sort
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, n_init=3)


# load detector
detector = load_yolo(cfg.MOT.DETECTOR)
detector.to(device)


# load static feature extractors
static_feature_models = {
    k: os.path.join(cfg.SYSTEM.ROOT_DIR, v) for d in cfg.MOT.STATIC_FEATURES for k, v in d.items()}

if len(cfg.MOT.STATIC_FEATURES) > 0:
    static_extractor = create_extractor(StaticFeatureExtractor, batch_size=cfg.MOT.STATIC_FEATURE_BATCHSIZE,
                                        model_paths_by_feature=static_feature_models)
else:
    static_extractor = None


# load input video
video_in = imageio.get_reader(
    os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.MOT.VIDEO))
video_meta = video_in.get_meta_data()
video_w, video_h = video_meta["size"]
video_frames = video_in.count_frames()


# load input mask if any
if cfg.MOT.DETECTION_MASK is not None:
    det_mask = Image.open(os.path.join(
        cfg.SYSTEM.ROOT_DIR, cfg.MOT.DETECTION_MASK))

    # convert mask to 1's and 0's (with some treshold, because dividing by 255
    # causes some black pixels if the mask is not exactly pixel perfect)
    det_mask = (np.array(det_mask) / 180).astype(np.uint8)
else:
    det_mask = None

# initialize output video
if cfg.MOT.VIDEO_OUTPUT:
    video_out = imageio.get_writer(os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.MOT.VIDEO_OUTPUT),
                                   format='FFMPEG', mode='I',
                                   fps=video_meta["fps"],
                                   codec=video_meta["codec"])

# initialize font
font = ImageFont.truetype(cfg.MOT.FONT, 13)

# initialize display
if cfg.MOT.SHOW:
    cv2.namedWindow("tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("tracking", 1280, 720)

# initialize color map
cmap = plt.get_cmap('hsv')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]


########################################
# Main tracking loop
########################################

fps_counter = FrameRateCounter()
tracklets = {}

for frame_num, frame in enumerate(video_in):
    res = detector(frame).xywh[0].cpu().numpy()

    # only bike, car, motorbike, bus, truck classes
    good_classes = [1, 2, 3, 5, 7]

    # detected boxes in cx,cy,w,h format
    boxes = [t[:4] for t in res]
    scores = [t[4] for t in res]
    classes = [t[5] for t in res]

    boxes = filter_boxes(boxes, scores, classes,
                         good_classes, 0.4, det_mask)

    boxes_tlwh = [[int(x - w / 2), int(y - h / 2), w, h]
                  for x, y, w, h in boxes]

    # TODO: non-max suppression?
    features = extractor(frame, boxes_tlwh)
    detections = [Detection(bbox, score, clname, feature)
                  for bbox, score, clname, feature in zip(boxes_tlwh, scores, classes, features)]

    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(
        boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    overlay = Image.fromarray(
        np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8))

    active_tracks = []
    active_track_bboxes = []
    for track in tracker.tracks:
        if track.track_id not in tracklets:
            tracklets[track.track_id] = Tracklet(track.track_id)
        if track.time_since_update > 1:
            continue

        bbox = track.to_tlbr()
        tx, ty, bx, by = list(map(lambda x: max(0, int(x)), bbox))
        w, h = int(bx - tx), int(by - ty)

        active_track_bboxes.append([tx, ty, w, h])
        active_tracks.append(track)

    if static_extractor is not None:
        static_features = static_extractor(frame, active_track_bboxes)
    else:
        static_features = [None] * len(active_tracks)

    for track, bbox, static_f in zip(active_tracks, active_track_bboxes, static_features):
        tracklet = tracklets[track.track_id]
        tx, ty, w, h = bbox
        bx, by = int(tx + w), int(ty + h)
        tracklet.update(frame_num, (tx, ty, w, h),
                        track.last_feature, static_f)

        color = colors[int(track.track_id) % len(colors)]
        color = [int(i * 255) for i in color]

        frame = draw_rectangle(frame, tx, ty, w, h, color, 1)
        overlay = annotate(overlay, str(track.track_id), static_f,
                           tx, by, color, font)

    mask = Image.fromarray((np.array(overlay) > 0).astype(np.uint8) * 255)
    frame_img = Image.fromarray(frame)
    frame_img.paste(overlay, mask=mask)

    # put frame number on the image
    put_text(frame_img, f"Frame {frame_num}", 0, 0, (255, 0, 0), font)

    frame = np.array(frame_img)

    if cfg.MOT.VIDEO_OUTPUT:
        video_out.append_data(frame)

    if cfg.MOT.SHOW:
        cv2.imshow("tracking", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    fps_counter.step()
    print("\rFrame: {}/{}, fps: {:.3f}".format(
        frame_num, video_frames, fps_counter.value()), end="")

if cfg.MOT.VIDEO_OUTPUT:
    video_out.close()

########################################
# Save results
########################################

# filter unconfirmed tracklets
final_tracks = list(tracklets.values())
final_tracks = list(filter(lambda track: len(
    track.frames) >= cfg.MOT.MIN_FRAMES, final_tracks))

if cfg.MOT.CSV_RESULT_PATH:
    save_path = os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.MOT.CSV_RESULT_PATH)
    save_tracklets_csv(final_tracks, save_path)

if cfg.MOT.RESULT_PATH:
    save_path = os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.MOT.RESULT_PATH)
    save_tracklets(final_tracks, save_path)
