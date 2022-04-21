from tkinter import ttk
from tkinter import *
import tkinter as tk
from PIL import Image, ImageOps, ImageTk, ImageDraw, ImageFont
import imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
from collections import deque

from mot.tracklet_processing import load_tracklets
from mot.static_features import FEATURES
from tools.preprocessing import extract_image_patch

parser = argparse.ArgumentParser(description="Reid annotation tool")
parser.add_argument("--max_per_class", type=int, default=20,
                    help="maximum number of images to associate with each id/class.")
parser.add_argument("--video", help="path to input video", required=True)
parser.add_argument("--tracklets", required=True,
                    help="tracklets pickle file")
parser.add_argument("--start_frame", default=0,
                    help="frame to start at")
args = parser.parse_args()


class TrackedVideo:
    def __init__(self, video_path, tracklets_path, start_frame=0, font="Hack-Regular.ttf",
                 keep_max_ended_tracks=10):
        self.video = imageio.get_reader(video_path)
        self.tracklets = load_tracklets(tracklets_path)
        self.font = ImageFont.truetype(font, 16)
        self.keep_max_ended_tracks = keep_max_ended_tracks
        cmap = plt.get_cmap('hsv')
        self.colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        self.frame_num = 0
        self.active_tracks = set()
        self.ended_tracks = []
        self.tracklets.sort(key=lambda tr: tr.frames[0])
        self.tracklets_ptr = 0

        while self.frame_num < start_frame:
            self.step_frame()
        self.ended_tracks = []

        # make sure a single value is predicted for static features
        for tracklet in self.tracklets:
            tracklet.predict_final_static_features()

    def step_frame(self):
        try:
            frame = self.video.get_next_data()
        except IndexError:
            return False

        while self.tracklets_ptr < len(self.tracklets) and \
                self.tracklets[self.tracklets_ptr].frames[0] == self.frame_num:
            track = self.tracklets[self.tracklets_ptr]
            track.images = []
            track.frame_ptr = 0
            self.active_tracks.add(self.tracklets_ptr)
            self.tracklets_ptr += 1

        inactivated_tracks = []
        for tridx in self.active_tracks:
            track = self.tracklets[tridx]
            if track.frames[track.frame_ptr] == self.frame_num:
                img = extract_image_patch(frame, track.bboxes[track.frame_ptr])
                img_pil = Image.fromarray(img)
                track.images.append(img_pil.resize((224, 224)))
                track.frame_ptr += 1
            if track.frame_ptr == len(track.frames):
                self.ended_tracks.append(tridx)
                inactivated_tracks.append(tridx)

        self.active_tracks = self.active_tracks.difference(inactivated_tracks)

        if len(self.ended_tracks) > self.keep_max_ended_tracks:
            for i in range(len(self.ended_tracks) - self.keep_max_ended_tracks):
                self.tracklets[self.ended_tracks[i]].images = None
            self.ended_tracks = self.ended_tracks[-self.keep_max_ended_tracks:]

        self.frame = frame
        self.frame_num += 1
        return True

    def get_current_frame(self):
        img = self.frame.copy()
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        tracks = list(self.active_tracks)
        for i in range(len(self.ended_tracks) - 1, -1, -1):
            if self.tracklets[self.ended_tracks[i]].frames[-1] == self.frame_num - 1:
                tracks.append(self.ended_tracks[i])
            else:
                break

        for tridx in tracks:
            track = self.tracklets[tridx]
            if track.frame_ptr > 0 and track.frames[track.frame_ptr - 1] == self.frame_num - 1:
                tx, ty, w, h = track.bboxes[track.frame_ptr - 1]
                bx, by = int(tx + w), int(ty + h)
                color = self.colors[int(track.track_id) % len(self.colors)]
                color = tuple(map(lambda x: int(255 * x), color))
                draw.rectangle([tx, ty, bx, by], outline=color)

                textcoords = draw.textbbox((tx, by), str(
                    track.track_id), font=self.font)
                if textcoords[3] >= img.size[1]:
                    txt_y = ty - (textcoords[3] - textcoords[1]) - 4
                else:
                    txt_y = by
                draw.text((tx, txt_y), str(track.track_id),
                          color, font=self.font)

        draw.text((0, 0), f"Frame {self.frame_num - 1}",
                  (255, 0, 0, 255), self.font)
        return img


class Annotator:
    def __init__(self, static_features=[], zones=False):
        self.tracklets = {}
        self.static_features = static_features
        self.zones = zones

    def add_annotations(self, tracklet, frame_idxes):
        self.tracklets[tracklet.track_id] = (tracklet, frame_idxes)

    def save_as_csv(self, path):
        columns = {
            "frame": [],
            "bbox_topleft_x": [],
            "bbox_topleft_y": [],
            "bbox_width": [],
            "bbox_height": [],
            "track_id": [],
        }

        for sf in self.static_features:
            columns[sf] = []

        if self.zones:
            columns["zone"] = []

        for trid, (tracklet, frame_idxes) in self.tracklets.items():
            frame_idxes = set(frame_idxes)
            track_id = tracklet.save_track_id
            for i in range(len(tracklet.frames)):
                if i in frame_idxes:
                    colnames = ["frame", "bbox_topleft_x", "bbox_topleft_y", "bbox_width",
                                "bbox_height", "track_id"]
                    x, y, w, h = tracklet.bboxes[i]
                    vals = [tracklet.frames[i], x, y, w, h, track_id]
                    for f, val in tracklet.static_features.items():
                        colnames.append(f)
                        vals.append(val)
                    if tracklet.zones:
                        colnames.append("zone")
                        vals.append(tracklet.zones[i])

                    for c, v in zip(colnames, vals):
                        columns[c].append(v)

        df = pd.DataFrame(columns)
        df.to_csv(path, index=False)


class TrackAnnotatorDialog:
    def __init__(self, parent, tracklet, num_images=40):
        self.top = Toplevel(parent)
        self.tracklet = tracklet
        self.num_images = min(num_images, len(tracklet.frames))

        # initialize buttons
        self.btn_frame = Frame(self.top)
        self.btn_ok = Button(
            self.btn_frame, text="Save tracklet", command=self.accept, background="green")
        self.btn_ok.grid(row=0, column=0, padx=3)
        self.btn_abort = Button(
            self.btn_frame, text="abort", command=self.cancel, background="red")
        self.btn_abort.grid(row=0, column=1, padx=3)
        self.btn_frame.pack(fill=X, expand=False)

        # initialize inputs
        self.input_frame = Frame(self.top)
        Label(self.input_frame, text="track id:").grid(row=0, column=0)
        self.track_id_input = Entry(self.input_frame)
        self.track_id_input.insert(0, str(tracklet.track_id))
        self.track_id_input.grid(row=0, column=1, padx=3)

        self.feature_inputs = {}
        for i, (feature, value) in enumerate(tracklet.static_features.items()):
            Label(self.input_frame, text=feature + ":").grid(
                row=0, column=2 + 2 * i)
            feature_var = StringVar(self.top)
            feature_var.set(FEATURES[feature][value])
            feature_choice = ttk.Combobox(
                self.input_frame, textvariable=feature_var, values=FEATURES[feature])
            feature_choice.grid(row=0, column=3 + 2 * i, padx=3)
            self.feature_inputs[feature] = feature_var

        self.input_frame.pack(fill=X, expand=False)

        # initialize the frame containing the images
        self.frame = Frame(self.top)
        self.n_rows = round(math.sqrt(self.num_images) * 9 / 16)
        self.n_cols = round(math.sqrt(self.num_images) * 16 / 9)
        while self.n_rows * self.n_cols < self.num_images:
            self.n_cols += 1

        max_width = (1280 - 4 * (self.n_cols + 1)) / self.n_cols
        max_height = (720 - 4 * (self.n_rows + 1)) / self.n_rows
        print(max_width, max_height, self.n_rows, self.n_cols)
        self.img_size = int(min(max_width, max_height))

        if len(tracklet.frames) <= self.num_images:
            self.current_idxes = list(range(len(tracklet.frames)))
        else:
            self.current_idxes = []
            for i in range(self.num_images):
                idx = int(i / self.num_images * len(tracklet.frames))
                self.current_idxes.append(idx)

        self.remaining_idxes = set(
            range(len(tracklet.frames))).difference(self.current_idxes)
        while len(self.remaining_idxes) > 0 and len(self.current_idxes) < self.num_images:
            q = next(iter(self.remaining_idxes))
            self.current_idxes.append(q)
            self.remaining_idxes.erase(q)

        self.img_labels = []
        for i, idx in enumerate(self.current_idxes):
            img = tracklet.images[idx]
            img = ImageOps.expand(img, border=2, fill="green")
            img = img.resize((self.img_size, self.img_size))
            photo_img = ImageTk.PhotoImage(img)
            img_label = Label(self.frame, image=photo_img)
            img_label.photo = photo_img
            img_label.selected = True
            img_label.bind("<Button-1>", lambda e,
                           idx=i: self.img_clicked(e, idx))
            img_label.grid(row=i // self.n_cols, column=i %
                           self.n_cols, padx=2, pady=2)
            self.img_labels.append(img_label)
        self.frame.pack(fill=BOTH, expand=True)

        # initialize modifier states
        self.shift = False
        self.last_click = 0
        self.accepted_idxes = None

    def select_img(self, idx):
        self.img_labels[idx].selected = True
        img = self.tracklet.images[self.current_idxes[idx]]
        img = ImageOps.expand(img, border=2, fill="green")
        img = img.resize((self.img_size, self.img_size))
        self.img_labels[idx].photo.paste(img)

    def unselect_img(self, idx):
        self.img_labels[idx].selected = False
        img = self.tracklet.images[self.current_idxes[idx]]
        img = ImageOps.expand(img, border=2, fill="red")
        img = img.resize((self.img_size, self.img_size))
        self.img_labels[idx].photo.paste(img)

    def img_clicked(self, event, idx):
        shift = bool(event.state & 1)
        if not shift:
            if self.img_labels[idx].selected:
                self.unselect_img(idx)
            else:
                self.select_img(idx)
        elif self.shift:
            r1, r2 = min(self.last_click, idx), max(self.last_click, idx) + 1
            all_selected = all(
                map(lambda x: x.selected, self.img_labels[r1:r2]))
            for i in range(r1, r2):
                if all_selected:
                    self.unselect_img(i)
                else:
                    self.select_img(i)
        self.shift = False if (self.shift and shift) else shift
        self.last_click = idx

    def accept(self):
        self.accepted_idxes = [idx for i, idx in enumerate(
            self.current_idxes) if self.img_labels[i].selected]
        self.top.destroy()

    def cancel(self):
        self.top.destroy()


class Main:
    def __init__(self, video_path, tracklets_path):
        self.video_path = video_path
        self.tracklets_path = tracklets_path
        self.root = Tk()
        self.style = ttk.Style()
        self.style.theme_use("breeze-dark")
        self.root.title("Video annotation")

        self.menubar = Menu(self.root)
        self.file_menu = Menu(self.menubar, tearoff=0)

        # initialize buttons
        self.button_row = Frame(self.root, borderwidth=2)
        btn_start = Button(self.button_row, text="Start",
                           command=self.step_until_new_tracklet)
        btn_start.pack(side=LEFT, padx=5, pady=2)
        self.tracklet_choice = Variable(self.root)
        self.tracklet_choice.set("")
        self.tracklet_options = ttk.Combobox(
            self.button_row, textvariable=self.tracklet_choice, values=[])
        self.tracklet_options.pack(side=LEFT, padx=5, pady=2)
        btn_annot = Button(self.button_row, text="Annotate track",
                           command=self.annotate_track)
        btn_annot.pack(side=LEFT, padx=5, pady=2)
        self.button_row.pack(fill=X)

        # initialize frame viewer
        self.frame_view = ttk.Frame(self.root, padding=2, borderwidth=2)
        self.photo = None
        self.frame_view.pack(fill=NONE, expand=True)

        # initialize video and annotation
        self.init_annotation()

        self.root.config(menu=self.menubar)
        self.root.mainloop()

    def init_annotation(self):
        self.video = TrackedVideo(self.video_path, self.tracklets_path)
        static_features = list(self.video.tracklets[0].static_features.keys())
        zones = bool(self.video.tracklets[0].zones)
        self.annotator = Annotator(static_features, zones)

    def update_tracklet_options(self):
        options = []
        for idx in reversed(self.video.ended_tracks):
            track_id = str(self.video.tracklets[idx].track_id)
            options.append(track_id)
        self.tracklet_options["values"] = options
        self.tracklet_choice.set(options[0])

    def update_frame_view(self):
        image = self.video.get_current_frame()
        image = image.resize((1280, 720))
        if self.photo is None:
            img = ImageTk.PhotoImage(image)
            self.photo = Label(self.frame_view, image=img)
            self.photo.image = img
            self.photo.pack()
        else:
            self.photo.image.paste(image)

    def step_until_new_tracklet(self):
        last_ended_track = None if len(
            self.video.ended_tracks) == 0 else self.video.ended_tracks[-1]
        step_valid = True
        while step_valid and (len(self.video.ended_tracks) == 0 or
                              last_ended_track == self.video.ended_tracks[-1]):
            step_valid = self.video.step_frame()
            self.update_frame_view()

        if step_valid:
            self.update_tracklet_options()
            return True
        return False

    def annotate_track(self):
        try:
            track_id = int(self.tracklet_choice.get())
            tracklet = list(filter(lambda t: t.track_id ==
                                   track_id, self.video.tracklets))[0]
        except (IndexError, ValueError):
            print("Invalid track_id chosen")
            return
        track_annot = TrackAnnotatorDialog(self.root, tracklet)
        self.root.wait_window(track_annot.top)
        if track_annot.accepted_idxes:
            self.annotator.add_annotations(
                tracklet, track_annot.accepted_idxes)
        print(track_annot.accepted_idxes)


main = Main(args.video, args.tracklets)
