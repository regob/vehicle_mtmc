from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps, ImageTk, ImageDraw, ImageFont
import imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import pickle

from mot.tracklet_processing import load_tracklets
from mot.attributes import STATIC_ATTRIBUTES
from tools.preprocessing import extract_image_patch

parser = argparse.ArgumentParser(description="Reid annotation tool")
parser.add_argument("--max_imgs_per_track", type=int, default=20,
                    help="maximum number of images per track to show.")
parser.add_argument("--video", help="path to input video", required=True)
parser.add_argument("--tracklets", required=True,
                    help="tracklets pickle file")
parser.add_argument("--start_frame", default=0, type=int,
                    help="frame to start at")
parser.add_argument("--reference_video", type=str, default="",
                    help="video to extract given tracks from for reference (to match track ids between videos)")
parser.add_argument("--reference_annot", type=str, default="",
                    help="refrence saved (pickled) tracklets from annotating the reference video")
parser.add_argument("--ref_time_ahead", type=float, default=0.0,
                    help="how many seconds the reference video is ahead of the video in time")
parser.add_argument("--ref_matching_zone", type=int, default=-1,
                    help="zone to check enter and leave framestamps in between reference video and current video.")
parser.add_argument("--unique_id_start", type=int, default=10000,
                    help="starting value for assigning new globally unique track_id's")
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
            tracklet.predict_final_static_attributes()

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

    def del_annotations(self, track_id):
        if track_id in self.tracklets:
            del self.tracklets[track_id]

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
                    for f, val in tracklet.static_attributes.items():
                        colnames.append(f)
                        vals.append(val)
                    if tracklet.zones:
                        colnames.append("zone")
                        vals.append(tracklet.zones[i])

                    for c, v in zip(colnames, vals):
                        columns[c].append(v)

        df = pd.DataFrame(columns)
        df.to_csv(path, index=False)

    def load_pickle(self, path):
        with open(path, "rb") as f:
            self.tracklets = pickle.load(f)
        for tr in self.tracklets.values():
            tr[0].images = None

    def save_as_pickle(self, path):
        for tr in self.tracklets.values():
            tr[0].images = None
        with open(path, "wb") as f:
            pickle.dump(self.tracklets, f)


class TrackAnnotatorDialog:
    def __init__(self, parent, tracklet, num_images=40, reference_tracklets=None, unique_id=10000):
        self.top = tk.Toplevel(parent)
        self.tracklet = tracklet
        self.num_images = min(num_images, len(tracklet.frames))
        self.unique_id = unique_id

        # initialize buttons
        self.btn_frame = ttk.Frame(self.top)
        self.btn_ok = ttk.Button(
            self.btn_frame, text="Save tracklet", command=self.accept)
        self.btn_ok.grid(row=0, column=0, padx=3)
        self.btn_abort = ttk.Button(
            self.btn_frame, text="abort", command=self.cancel)
        self.btn_abort.grid(row=0, column=1, padx=3)
        self.btn_unique_id = ttk.Button(
            self.btn_frame, text="unique id", command=self._assign_unique_id)
        self.btn_unique_id.grid(row=0, column=2, padx=3)
        self.btn_frame.pack(fill=tk.X, expand=False, padx=2, pady=2)

        # add button keybinds
        self.top.bind("<q>", lambda e: self.cancel())
        self.top.bind("<w>", lambda e: self.accept())
        self.top.bind("<u>", lambda e: self._assign_unique_id())

        # initialize inputs
        self.input_frame = ttk.Frame(self.top)
        ttk.Label(self.input_frame, text="track id:").grid(row=0, column=0)
        self.track_id_input = ttk.Entry(self.input_frame)
        self.track_id_input.insert(0, str(tracklet.track_id))
        self.track_id_input.grid(row=0, column=1, padx=3)

        self.feature_inputs = {}
        for i, (feature, value) in enumerate(tracklet.static_attributes.items()):
            ttk.Label(self.input_frame, text=feature + ":").grid(
                row=0, column=2 + 2 * i)
            feature_var = tk.StringVar(self.top)
            feature_var.set(STATIC_ATTRIBUTES[feature][value])
            feature_choice = ttk.Combobox(
                self.input_frame, textvariable=feature_var, values=STATIC_ATTRIBUTES[feature], state="readonly")
            feature_choice.grid(row=0, column=3 + 2 * i, padx=3)
            self.feature_inputs[feature] = feature_var

        self.input_frame.pack(fill=tk.X, expand=False, pady=2, padx=2)

        # initialize the frame containing the images
        self.frame = ttk.Frame(self.top)
        self.n_rows = round(math.sqrt(self.num_images) * 9 / 16)
        self.n_cols = round(math.sqrt(self.num_images) * 16 / 9)
        while self.n_rows * self.n_cols < self.num_images:
            self.n_cols += 1

        max_width = (1280 - 4 * (self.n_cols + 1)) / self.n_cols
        max_height = (720 - 4 * (self.n_rows + 1)) / self.n_rows
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
            img = ImageOps.expand(img, border=4, fill="green")
            img = img.resize((self.img_size, self.img_size))
            photo_img = ImageTk.PhotoImage(img)
            img_label = ttk.Label(self.frame, image=photo_img)
            img_label.photo = photo_img
            img_label.selected = True
            img_label.bind("<Button-1>", lambda e,
                           idx=i: self.img_clicked(e, idx))
            img_label.grid(row=i // self.n_cols, column=i %
                           self.n_cols, padx=2, pady=2)
            self.img_labels.append(img_label)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # initialize frame of reference tracklets
        if reference_tracklets:
            self.reference_frame = ttk.Frame(self.top)
            ref_width = min(
                100, (1280 - 4 * len(reference_tracklets)) / len(reference_tracklets))
            for i, track in enumerate(reference_tracklets):
                frame = ttk.Frame(self.reference_frame)
                photo_img = ImageTk.PhotoImage(
                    track.image.resize((ref_width, ref_width)))
                img_label = ttk.Label(frame, image=photo_img)
                img_label.photo = photo_img
                img_label.bind("<Button-1>",
                               lambda _, track=track: self._choose_ref_track(track))
                img_label.grid(row=0, column=0)
                ttk.Label(frame, text=str(
                    track.save_track_id)).grid(row=1, column=0)
                frame.grid(row=0, column=i, padx=2, pady=2)
            self.reference_frame.pack()

        # initialize modifier states
        self.shift = False
        self.last_click = 0
        self.accepted_idxes = None

    def _assign_unique_id(self):
        self.set_id_input(str(self.unique_id))

    def _choose_ref_track(self, track):
        self.set_id_input(str(track.save_track_id))
        for f, val in track.static_attributes.items():
            self.feature_inputs[f].set(STATIC_ATTRIBUTES[f][val])

    def set_id_input(self, text):
        self.track_id_input.delete(0, "end")
        self.track_id_input.insert(0, text)
        self.frame.focus_set()

    def select_img(self, idx):
        self.img_labels[idx].selected = True
        img = self.tracklet.images[self.current_idxes[idx]]
        img = ImageOps.expand(img, border=4, fill="green")
        img = img.resize((self.img_size, self.img_size))
        self.img_labels[idx].photo.paste(img)

    def unselect_img(self, idx):
        self.img_labels[idx].selected = False
        img = self.tracklet.images[self.current_idxes[idx]]
        img = ImageOps.expand(img, border=4, fill="red")
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
        self.tracklet.save_track_id = int(self.track_id_input.get())
        for f, val in self.feature_inputs.items():
            self.tracklet.static_attributes[f] = STATIC_ATTRIBUTES[f].index(
                val.get())
        self.top.destroy()

    def cancel(self):
        self.top.destroy()


class Main:
    def __init__(self, video_path, tracklets_path, start_frame=0, reference_tracks=None,
                 ref_time_ahead=0.0, ref_matching_zone=-1, ref_max_time_gap=6.0, ref_video_fps=None,
                 unique_id_start=10000, max_imgs_per_track=20):
        self.video_path = video_path
        self.start_frame = start_frame
        self.tracklets_path = tracklets_path
        self.reference_tracks = reference_tracks
        self.ref_time_ahead = ref_time_ahead
        self.ref_matching_zone = ref_matching_zone
        self.ref_max_time_gap = ref_max_time_gap
        self.ref_video_fps = ref_video_fps
        self.next_unique_id = unique_id_start
        self.max_imgs_per_track = max_imgs_per_track
        self.root = tk.Tk()
        try:
            # self.root.tk.call("source", "~/.themes/azure/azure.tcl")
            # self.root.tk.call("set_theme", "dark")
            pass
        except:
            print("Setting azure theme failed.")
            pass
        # self.style = ttk.Style()
        # self.style.theme_use("breeze")
        self.root.title("Video annotation")

        self.menubar = tk.Menu(self.root)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)

        # initialize buttons
        self.button_row = ttk.Frame(self.root, borderwidth=2)
        btn_start = ttk.Button(self.button_row, text="Step",
                               command=self.step_until_new_tracklet)
        btn_start.pack(side=tk.LEFT, padx=5, pady=2)
        self.tracklet_choice = tk.Variable(self.root)
        self.tracklet_choice.set("")
        self.tracklet_options = ttk.Combobox(
            self.button_row, textvariable=self.tracklet_choice, values=[])
        self.tracklet_options.bind("<<ComboboxSelected>>",
                                   lambda _: self.button_row.focus_set())
        self.tracklet_options.pack(side=tk.LEFT, padx=5, pady=2)
        btn_annot = ttk.Button(self.button_row, text="Annotate track",
                               command=self.annotate_track)
        btn_annot.pack(side=tk.LEFT, padx=5, pady=2)

        btn_delete = ttk.Button(self.button_row, text="Delete track",
                                command=self.delete_track)
        btn_delete.pack(side=tk.LEFT, padx=5, pady=2)

        btn_save_csv = ttk.Button(self.button_row, text="Save to csv",
                                  command=self.save_csv)
        btn_save_csv.pack(side=tk.LEFT, padx=5, pady=2)
        btn_save_pickle = ttk.Button(self.button_row, text="Save to pickle",
                                     command=self.save_pickle)
        btn_save_pickle.pack(side=tk.LEFT, padx=5, pady=2)
        btn_load_pickle = ttk.Button(self.button_row, text="Load pickle",
                                     command=self.load_pickle)
        btn_load_pickle.pack(side=tk.LEFT, padx=5, pady=2)
        self.button_row.pack(fill=tk.X)

        # initialize frame viewer
        self.frame_view = ttk.Frame(self.root, padding=2, borderwidth=2)
        self.photo = None
        self.frame_view.pack(fill=tk.NONE, expand=True)

        # initialize status bar
        self.statusbar = ttk.Label(self.root, text="",
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        # initialize video and annotation
        self.init_annotation()
        self.last_ended_track = None

        # add keybinds
        self.root.bind("<space>", lambda _: self.step_until_new_tracklet())
        self.root.bind("<a>", lambda _: self.annotate_track())
        self.root.bind("<Control-s>", lambda _: self.save_csv())

        self.root.config(menu=self.menubar)
        self.root.mainloop()

    def init_annotation(self):
        self.video = TrackedVideo(
            self.video_path, self.tracklets_path, self.start_frame, "../../assets/Hack-Regular.ttf")
        static_features = list(
            self.video.tracklets[0].static_attributes.keys())
        zones = bool(self.video.tracklets[0].zones)
        self.annotator = Annotator(static_features, zones)

    def update_tracklet_options(self):
        options = []
        for idx in reversed(self.video.ended_tracks):
            track_id = str(self.video.tracklets[idx].track_id)
            options.append(track_id)
        self.tracklet_options["values"] = options
        self.tracklet_choice.set(str(self.video.tracklets[self.last_ended_track].track_id))

    def update_frame_view(self):
        image = self.video.get_current_frame()
        image = image.resize((1280, 720))
        if self.photo is None:
            img = ImageTk.PhotoImage(image)
            self.photo = ttk.Label(self.frame_view, image=img)
            self.photo.image = img
            self.photo.pack()
            self.photo.bind(
                "<Button-1>", lambda _: self.photo.focus_set())
        else:
            self.photo.image.paste(image)

    def step_until_new_tracklet(self):
        step_valid = True
        if len(self.video.ended_tracks) == 0 or self.last_ended_track == self.video.ended_tracks[-1]:
            while step_valid and (len(self.video.ended_tracks) == 0 or
                                  self.last_ended_track == self.video.ended_tracks[-1]):
                step_valid = self.video.step_frame()
            self.update_frame_view()

        try:
            idx = self.video.ended_tracks.index(self.last_ended_track) + 1
        except ValueError:
            idx = 0
        if idx < len(self.video.ended_tracks):
            self.last_ended_track = self.video.ended_tracks[idx]

        if step_valid:
            self.update_tracklet_options()
            return True
        self.statusbar["text"] = "Video ended."
        return False

    def annotate_track(self):
        try:
            track_id = int(self.tracklet_choice.get())
            tracklet = list(filter(lambda t: t.track_id ==
                                   track_id, self.video.tracklets))[0]
        except (IndexError, ValueError):
            self.statusbar["text"] = "Invalid track id."
            return
        if self.reference_tracks is not None:

            if self.ref_matching_zone:
                enter, leave = tracklet.zone_enter_leave_frames(
                    self.ref_matching_zone)
            else:
                enter, leave = tracklet.frames[0], tracklet.frames[-1]
            curr_video_fps = float(self.video.video.get_meta_data()["fps"])
            enter = enter / curr_video_fps
            leave = leave / curr_video_fps

            if self.ref_time_ahead:
                enter -= self.ref_time_ahead
                leave -= self.ref_time_ahead

            mean_f = np.zeros_like(tracklet.features[0])
            for f in tracklet.features:
                mean_f += f
            mean_f = mean_f / np.linalg.norm(mean_f)
            reference_dists = []

            for tr_id, (ref_track, _) in self.reference_tracks.items():
                ref_enter, ref_leave = ref_track.zone_enter_leave_frames(
                    self.ref_matching_zone)
                ref_enter = ref_enter / self.ref_video_fps
                ref_leave = ref_leave / self.ref_video_fps
                if (enter < 0 and ref_enter >= 0) or (
                        ref_enter < 0 and enter >= 0):
                    continue

                if ref_leave + self.ref_max_time_gap < enter or \
                   leave + self.ref_max_time_gap < ref_enter:
                    continue

                ref_mean_f = np.zeros_like(mean_f)
                for f in ref_track.features:
                    ref_mean_f += f
                ref_mean_f /= np.linalg.norm(ref_mean_f)
                reference_dists.append(
                    (1 - np.dot(mean_f, ref_mean_f), ref_track))

            reference_dists.sort(key=lambda x: x[0])
            ref_tracks = list(map(lambda t: t[1], reference_dists[:12]))
        else:
            ref_tracks = None

        track_annot = TrackAnnotatorDialog(
            self.root, tracklet, self.max_imgs_per_track, ref_tracks, unique_id=self.next_unique_id)
        self.root.wait_window(track_annot.top)
        if track_annot.accepted_idxes:
            self.annotator.add_annotations(
                tracklet, track_annot.accepted_idxes)
            if tracklet.save_track_id == self.next_unique_id:
                self.next_unique_id += 1
            self.statusbar["text"] = f"Track {tracklet.save_track_id} added ({len(track_annot.accepted_idxes)} images)."
        else:
            self.statusbar["text"] = f"Track {tracklet.track_id} aborted."

    def delete_track(self):
        try:
            track_id = int(self.tracklet_choice.get())
            self.annotator.del_annotations(track_id)
            self.statusbar["text"] = "Track {} annotations deleted.".format(
                track_id)
            assert track_id not in self.annotator.tracklets
        except ValueError:
            self.statusbar["text"] = "Error: deleting track unsuccessful."

    def save_csv(self):
        path = filedialog.asksaveasfilename()
        if path:
            self.annotator.save_as_csv(path)
            self.statusbar["text"] = "Annotations saved as csv."
        else:
            self.statusbar["text"] = "Invalid path chosen"

    def save_pickle(self):
        path = filedialog.asksaveasfilename()
        if path:
            self.annotator.save_as_pickle(path)
            self.statusbar["text"] = "Annotations saved."
        else:
            self.statusbar["text"] = "Invalid path chosen"

    def load_pickle(self):
        path = filedialog.askopenfilename()
        if path:
            self.annotator.load_pickle(path)
            self.statusbar["text"] = "Annotations loaded"
            max_id = max(
                map(lambda tr: tr[0].save_track_id, self.annotator.tracklets.values()))
            if max_id > self.next_unique_id:
                self.next_unique_id = max_id + 1
        else:
            self.statusbar["text"] = "Invalid path chosen"


def load_reference_tracks(video_path, annot_path):
    with open(annot_path, "rb") as f:
        track_dict = pickle.load(f)
    biggest_bbox = {}
    for _, tr in track_dict.items():
        track, accepted_idxes = tr
        best, best_size = 0, 0
        for idx in accepted_idxes:
            x, y, w, h = track.bboxes[idx]
            if w * h > best_size:
                best, best_size = idx, w * h
        biggest_bbox[track.track_id] = (track.frames[best], track.bboxes[best])

    biggest_bbox = list(biggest_bbox.items())
    biggest_bbox.sort(key=lambda x: x[1][0])

    ptr = 0
    video = imageio.get_reader(video_path)
    for frame_idx, frame in enumerate(video):
        while ptr < len(biggest_bbox) and biggest_bbox[ptr][1][0] == frame_idx:
            track_id, (_, bbox) = biggest_bbox[ptr]
            img = Image.fromarray(extract_image_patch(frame, bbox))
            img = img.resize((224, 224))
            track_dict[track_id][0].image = img
            ptr += 1

        if ptr >= len(biggest_bbox):
            break

    return track_dict


if args.reference_video and args.reference_annot:
    reference_tracks = load_reference_tracks(args.reference_video,
                                             args.reference_annot)
    vid = imageio.get_reader(args.reference_video)
    ref_video_fps = float(vid.get_meta_data()["fps"])
else:
    print("Reference tracks won't be loaded, not enough arguments given.")
    reference_tracks = None
    ref_video_fps = None

main = Main(args.video,
            args.tracklets,
            start_frame=args.start_frame,
            reference_tracks=reference_tracks,
            ref_time_ahead=args.ref_time_ahead,
            ref_matching_zone=args.ref_matching_zone,
            ref_video_fps=ref_video_fps,
            unique_id_start=args.unique_id_start,
            max_imgs_per_track=args.max_imgs_per_track)
