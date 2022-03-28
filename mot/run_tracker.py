import torch
import numpy as np
import os
import sys
import time
from PIL import Image, ImageDraw, ImageFont
import imageio
import argparse

import matplotlib.pyplot as plt
import cv2

# deep sort imports
from mot.deep_sort import preprocessing, nn_matching
from mot.deep_sort.detection import Detection
from mot.deep_sort.tracker import Tracker

from reid.feature_extractor import create_extractor
from reid.vehicle_reid.load_model import load_model_from_opts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Multi-object tracker on a video.")
    parser.add_argument("video", help="path to video to run on")
    parser.add_argument("--iou", default=0.45, type=float, help="iou treshold")
    parser.add_argument("--score", default=0.5,
                        type=float, help="score treshold")
    parser.add_argument("--show", action="store_true",
                        help="show video output")
    parser.add_argument("--video_output", default=None,
                        type=str, help="output video path")
    parser.add_argument("--reid_model_opts", default="../reid/vehicle_reid/model/resnet_ibn/opts.yaml",
                        help="opts.yaml for the reid model to use")
    parser.add_argument(
        "--reid_model_ckpt", default="../reid/vehicle_reid/model/resnet_ibn/net_11.pth")
    parser.add_argument("--detector", default="yolov5s",
                        help="object detector model")
    parser.add_argument("--cpu", action="store_true", help="run on cpu")
    parser.add_argument("--font", default="/usr/share/fonts/truetype/NimbusRomNo9L-Regu.ttf",
                        help="Font to use for text subscriptions")
    return parser.parse_args()


def draw_rectangle(img_np, tx, ty, w, h, color, width):
    """ Draw a colored rectangle to a numpy image. The top left corner is at (x,y), the rect has
    a width of w and a height of h. 'width' is the border width of the rectangle."""

    color = np.array(color)

    # bottom right coordinates
    bx, by = int(tx + w), int(ty + h)

    ranges = np.array([[int(ty - width / 2), int(ty + width / 2), tx, bx],
                       [int(by - width / 2), int(by + width / 2), tx, bx],
                       [ty, by, int(tx - width / 2), int(tx + width / 2)],
                       [ty, by, int(bx - width / 2), int(bx + width / 2)]])

    ranges[np.where(ranges < 0)] = 0

    for r in ranges:
        img_np[r[0]:r[1], r[2]:r[3]] = color
    return img_np


def put_text(img_pil, text, x, y, color, font):
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, (color[0], color[1], color[2], 255), font=font)
    return img_pil


def run_tracker(args):
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    device = torch.device(
        "cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")

    # initialize reid model
    reid_model = load_model_from_opts(args.reid_model_opts, ckpt=args.reid_model_ckpt,
                                      remove_classifier=True).to(device)
    reid_model.eval()
    extractor = create_extractor(reid_model)

    # initialize deep_sort
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    #   detector = torch.load("../detection/{}.pt".format(args.detector))
    #   detector = detector["model"]
    detector = torch.hub.load("ultralytics/yolov5", args.detector)
    detector.to(device)

    video_in = imageio.get_reader(args.video)
    video_meta = video_in.get_meta_data()
    video_w, video_h = video_meta["size"]

    if args.video_output:
        video_out = imageio.get_writer(args.video_output,
                                       format='FFMPEG', mode='I',
                                       fps=video_meta["fps"],
                                       codec=video_meta["codec"])

    # initialize font
    font = ImageFont.truetype(args.font, 20)

    # initialize display
    if args.show:
        cv2.namedWindow("tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("tracking", 1280, 720)

    for frame_num, frame in enumerate(video_in):
        res = detector(frame).xywh[0].cpu().numpy()

        # only bike, car, motorbike, bus, truck classes
        good_classes = [1, 2, 3, 5, 7]
        res = list(filter(lambda t: int(t[5]) in good_classes, res))

        # detected boxes in x,y,w,h format
        boxes = [t[:4] for t in res]
        scores = [t[4] for t in res]
        classes = [t[5] for t in res]

        boxes_tlwh = [[max(0, int(x - w / 2)), max(0, int(y - h / 2)), w, h]
                      for x, y, w, h in boxes]

        # TODO: non-max suppression?

        features = extractor(frame, boxes)
        detections = [Detection(bbox, score, clname, feature)
                      for bbox, score, clname, feature in zip(boxes_tlwh, scores, classes, features)]

        # initialize color map
        cmap = plt.get_cmap('hsv')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

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
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            tx, ty, w, h = int(bbox[0]), int(
                bbox[1]), int(bbox[2]), int(bbox[3])

            color = colors[int(track.track_id) % len(colors)]
            color = [int(i * 255) for i in color]

            frame = draw_rectangle(frame, tx, ty, w, h, color, 2)
            overlay = put_text(overlay, str(track.track_id),
                               tx, int(ty + h), color, font)

        mask = Image.fromarray((np.array(overlay) > 0).astype(np.uint8) * 255)
        frame_img = Image.fromarray(frame)
        frame_img.paste(overlay, mask=mask)

        # put frame number on the image
        put_text(frame_img, f"Frame {frame_num}", 0, 0, (255, 0, 0), font)

        frame = np.array(frame_img)

        if args.video_output:
            video_out.append_data(frame)

        if args.show:
            cv2.imshow("tracking", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        print("\rFrame: {}".format(frame_num), end="")

    if args.video_output:
        video_out.close()


if __name__ == "__main__":
    args = parse_args()
    run_tracker(args)
