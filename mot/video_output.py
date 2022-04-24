import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import imageio
import argparse

try:
    import cv2
except ImportError as e:
    print("cv2 import failed, on-the-fly video output not available: {}".format(e))


from mot.static_features import FEATURES
from mot.tracklet import Tracklet


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


class Video:
    def __init__(self, font):
        cmap = plt.get_cmap('hsv')
        self.colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        self.font = ImageFont.truetype(font, 13)
        self.frame_num = 0

    def render_tracks(self, frame, track_ids, track_bboxes, static_features):
        overlay = Image.fromarray(
            np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8))
        for track_id, bbox, static_f in zip(track_ids, track_bboxes, static_features):
            tx, ty, w, h = bbox
            bx, by = int(tx + w), int(ty + h)
            color = self.colors[int(track_id) % len(self.colors)]
            color = [int(i * 255) for i in color]
            frame = draw_rectangle(frame, tx, ty, w, h, color, 1)
            overlay = annotate(overlay, str(track_id), static_f,
                               tx, by, color, self.font)

        mask = Image.fromarray((np.array(overlay) > 0).astype(np.uint8) * 255)
        frame_img = Image.fromarray(frame)
        frame_img.paste(overlay, mask=mask)

        put_text(frame_img, f"Frame {self.frame_num}",
                 0, 0, (255, 0, 0), self.font)
        self.frame_num += 1

        return np.array(frame_img)


class DisplayVideo(Video):
    def __init__(self, font, width=1280, height=720):
        super().__init__(font)
        cv2.namedWindow("tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("tracking", width, height)

    def update(self, frame, track_ids, bboxes, static_features):
        frame = self.render_tracks(frame, track_ids, bboxes, static_features)
        cv2.imshow("tracking", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow("tracking")


class FileVideo(Video):
    def __init__(self, font, save_path, fps, codec, format="FFMPEG", mode="I"):
        super().__init__(font)
        self.video = imageio.get_writer(save_path, format=format, mode=mode,
                                        fps=fps, codec=codec)

    def update(self, frame, track_ids, bboxes, static_features):
        frame = self.render_tracks(frame, track_ids, bboxes, static_features)
        self.video.append_data(frame)

    def close(self):
        self.video.close()


def annotate_video_with_tracklets(input_path, output_path, tracklets, font="Hack-Regular.ttf"):
    video_in = imageio.get_reader(input_path)
    video_meta = video_in.get_meta_data()
    video_out = FileVideo(
        font, output_path, video_meta["fps"], video_meta["codec"])

    tracklets = sorted(tracklets, key=lambda tr: tr.frames[0])
    active_tracks = {}
    nxt_track = 0

    for frame_idx, frame in enumerate(video_in):
        while nxt_track < len(tracklets) and tracklets[nxt_track].frames[0] == frame_idx:
            active_tracks[nxt_track] = 0
            nxt_track += 1

        track_ids, bboxes, static_f = [], [], []
        ended_tracks = []
        incr_tracks = []

        # gather info for the current frame
        for track_idx, ptr in active_tracks.items():
            track = tracklets[track_idx]
            static_refined = isinstance(
                next(iter(track.static_features.values())), int)

            if track.frames[ptr] == frame_idx:
                track_ids.append(track.track_id)
                bboxes.append(track.bboxes[ptr])

                if static_refined:
                    static_f.append(track.static_features)
                else:
                    static_f.append({k: v[ptr]
                                     for k, v in track.static_features.items()})

                if ptr >= len(track.frames) - 1:
                    ended_tracks.append(track_idx)
                else:
                    incr_tracks.append(track_idx)

        for track_idx in ended_tracks:
            del active_tracks[track_idx]
        for track_idx in incr_tracks:
            active_tracks[track_idx] += 1

        video_out.update(frame, track_ids, bboxes, static_f)

    video_out.close()
