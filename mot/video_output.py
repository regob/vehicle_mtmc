import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import imageio
import argparse

try:
    import cv2
except ImportError as e:
    print("cv2 import failed, on-the-fly video output not available: {}".format(e))


from mot.attributes import get_attribute_value
from mot.tracklet_processing import load_tracklets
from tools import log


def put_text(img_pil, text, x, y, color, font):
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, (color[0], color[1], color[2],
                             255), font=font)
    return img_pil


def annotate(img_pil, id_label, attributes, tx, ty, bx, by, color, font):
    """ Put the id label and the features as text below or above of a bounding box. """

    draw = ImageDraw.Draw(img_pil, "RGBA")
    draw.rectangle([tx, ty, bx, by], outline=color, width=3)
    text = [id_label] + [f"{k}: {get_attribute_value(k, v)}" for k,
                         v in attributes.items()]
    text = "\n".join(text)

    textcoords = draw.multiline_textbbox((tx, by), text, font=font)

    # if the annotation below the box stretches out of the image, put it above
    if textcoords[3] >= img_pil.size[1]:
        txt_y = ty - (textcoords[3] - textcoords[1]) - 4
    else:
        txt_y = by

    # draw rectangle in the background
    coords = draw.multiline_textbbox((tx, txt_y), text, font=font)
    # add some padding
    textcoords = (coords[0] - 2, coords[1] - 2, coords[2] + 2, coords[3] + 2)
    draw.rectangle(textcoords, fill=color)

    # draw the text finally
    draw.multiline_text((tx, txt_y), text, (0, 0, 0), font=font)
    return img_pil


class Video:
    def __init__(self, font, fontsize=13):
        cmap = plt.get_cmap("Set2")
        self.colors = [cmap(i)[:3] for i in range(cmap.N)]
        cmap2 = plt.get_cmap("hsv")
        for i in np.linspace(0.1, 0.5, 7):
            self.colors.append(cmap2(i)[:3])
        self.HASH_Q = int(1e9 + 7)

        try:
            self.font = ImageFont.truetype(font, fontsize)
        except OSError:
            log.error(f"Video: Font {font} cannot be loaded, using PIL default font.")
            self.font = ImageFont.load_default()
        self.frame_font = ImageFont.truetype(font, 18)
        self.frame_num = 0

    def render_tracks(self, frame, track_ids, track_bboxes, attributes):
        overlay = Image.fromarray(
            np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8), "RGBA")
        for track_id, bbox, attrib in zip(track_ids, track_bboxes, attributes):
            tx, ty, w, h = bbox
            bx, by = int(tx + w), int(ty + h)
            color = self.colors[(self.HASH_Q * int(track_id)) % len(self.colors)]
            color = tuple(int(i * 255) for i in color)

            overlay = annotate(overlay, str(track_id), attrib,
                               tx, ty, bx, by, color, self.font)

        mask = Image.fromarray((np.array(overlay) > 0).astype(np.uint8) * 192)
        frame_img = Image.fromarray(frame)
        frame_img.paste(overlay, mask=mask)

        put_text(frame_img, f"Frame {self.frame_num}",
                 0, 0, (255, 0, 0), self.frame_font)
        self.frame_num += 1

        return np.array(frame_img)


class DisplayVideo(Video):
    def __init__(self, font, width=1280, height=720, fontsize=13):
        super().__init__(font, fontsize=fontsize)
        cv2.namedWindow("tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("tracking", width, height)

    def update(self, frame, track_ids, bboxes, attributes):
        frame = self.render_tracks(frame, track_ids, bboxes, attributes)
        cv2.imshow("tracking", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow("tracking")


class FileVideo(Video):
    def __init__(self, font, save_path, fps, codec, format="FFMPEG", mode="I", fontsize=13):
        super().__init__(font, fontsize=fontsize)
        self.video = imageio.get_writer(save_path, format=format, mode=mode,
                                        fps=fps, codec=codec, macro_block_size=8)

    def update(self, frame, track_ids, bboxes, attributes):
        frame = self.render_tracks(frame, track_ids, bboxes, attributes)
        self.video.append_data(frame)

    def close(self):
        self.video.close()


def annotate_video_with_tracklets(input_path, output_path, tracklets, font="Hack-Regular.ttf",
                                  fontsize=13):
    video_in = imageio.get_reader(input_path)
    video_meta = video_in.get_meta_data()
    video_out = FileVideo(
        font, output_path, video_meta["fps"], video_meta["codec"], fontsize=fontsize)

    tracklets = sorted(tracklets, key=lambda tr: tr.frames[0])
    active_tracks = {}
    nxt_track = 0

    for frame_idx, frame in enumerate(video_in):
        while nxt_track < len(tracklets) and tracklets[nxt_track].frames[0] == frame_idx:
            active_tracks[nxt_track] = 0
            nxt_track += 1

        track_ids, bboxes, attribs = [], [], []
        ended_tracks = []
        incr_tracks = []

        # gather info for the current frame
        for track_idx, ptr in active_tracks.items():
            track = tracklets[track_idx]

            try:
                static_refined = isinstance(
                    next(iter(track.static_attributes.values())), int)
            except StopIteration:
                static_refined = True

            if track.frames[ptr] == frame_idx:
                track_ids.append(track.track_id)
                bboxes.append(track.bboxes[ptr])

                attr = {}
                for k, v in track.static_attributes.items():
                    if static_refined:
                        attr[k] = v
                    else:
                        attr[k] = v[ptr]
                for k, v in track.dynamic_attributes.items():
                    attr[k] = v[ptr]
                attribs.append(attr)

                if ptr >= len(track.frames) - 1:
                    ended_tracks.append(track_idx)
                else:
                    incr_tracks.append(track_idx)

        for track_idx in ended_tracks:
            del active_tracks[track_idx]
        for track_idx in incr_tracks:
            active_tracks[track_idx] += 1

        video_out.update(frame, track_ids, bboxes, attribs)

    video_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="annotate video with tracklets")
    parser.add_argument("input_video", help="video to annotate")
    parser.add_argument("output_video", help="video output path")
    parser.add_argument(
        "tracklets", help="pickle file containing the tracklets")
    parser.add_argument("--fontsize", default=13, type=int,
                        help="font size for the annotation")
    args = parser.parse_args()

    tracklets = load_tracklets(args.tracklets)
    annotate_video_with_tracklets(args.input_video, args.output_video, tracklets,
                                  fontsize=args.fontsize)
