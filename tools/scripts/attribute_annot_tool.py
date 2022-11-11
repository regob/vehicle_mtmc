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
import os

from mot.tracklet_processing import load_tracklets
from mot.attributes import STATIC_ATTRIBUTES
from tools.preprocessing import extract_image_patch

parser = argparse.ArgumentParser(description="Reid annotation tool")
parser.add_argument("--data_dir", required=True)
parser.add_argument("--df_path", required=True)
args = parser.parse_args()

DATA_DIR = args.data_dir


class Annotator:
    def __init__(self, static_features=[]):
        self.tracklets = {}
        self.static_features = static_features

    def add_annotations(self, track_id, attributes):
        self.tracklets[track_id] = attributes

    def del_annotations(self, track_id):
        if track_id in self.tracklets:
            del self.tracklets[track_id]

    def save_as_csv(self, path):
        columns = {
            "track_id": [],
        }

        for sf in self.static_features:
            columns[sf] = []

        for trid, static in self.tracklets.items():
            columns["track_id"].append(trid)
            for k, v in static.items():
                columns[k].append(v)

        df = pd.DataFrame(columns)
        df.to_csv(path, index=False)


class TrackAnnotatorDialog:
    def __init__(self, parent, df, track_id, num_images=20, init_features=None):
        self.top = tk.Toplevel(parent)
        self.df = df[df["id"] == track_id]
        self.track_id = track_id
        self.num_images = min(num_images, len(self.df))

        # initialize buttons
        self.btn_frame = ttk.Frame(self.top)
        self.btn_ok = ttk.Button(
            self.btn_frame, text="Save tracklet", command=self.accept)
        self.btn_ok.grid(row=0, column=0, padx=3)
        self.btn_abort = ttk.Button(
            self.btn_frame, text="abort", command=self.cancel)
        self.btn_abort.grid(row=0, column=1, padx=3)
        self.btn_frame.pack(fill=tk.X, expand=False, padx=2, pady=2)

        # add button keybinds
        self.top.bind("<q>", lambda e: self.cancel())
        self.top.bind("<w>", lambda e: self.accept())

        # initialize inputs
        self.input_frame = ttk.Frame(self.top)
        self.feature_inputs = {}
        for i, feature in enumerate(STATIC_ATTRIBUTES.keys()):
            ttk.Label(self.input_frame, text=feature + ":").grid(
                row=0, column=2 * i)
            feature_var = tk.StringVar(self.top)
            if init_features is None:
                feature_var.set(STATIC_ATTRIBUTES[feature][0])
            else:
                feature_var.set(STATIC_ATTRIBUTES[feature][init_features[feature]])
            feature_choice = ttk.Combobox(
                self.input_frame, textvariable=feature_var, values=STATIC_ATTRIBUTES[feature], state="readonly")
            feature_choice.grid(row=0, column=1 + 2 * i, padx=3)
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

        if len(self.df) <= self.num_images:
            self.current_idxes = list(range(len(self.df)))
        else:
            self.current_idxes = []
            for i in range(self.num_images):
                idx = int(i / self.num_images * len(self.df))
                self.current_idxes.append(idx)

        self.img_labels = []
        for i, idx in enumerate(self.current_idxes):
            img = Image.open(os.path.join(DATA_DIR, self.df.iloc[idx]["path"]))
            img = img.resize((self.img_size, self.img_size))
            photo_img = ImageTk.PhotoImage(img)
            img_label = ttk.Label(self.frame, image=photo_img)
            img_label.photo = photo_img
            img_label.selected = True
            img_label.grid(row=i // self.n_cols, column=i %
                           self.n_cols, padx=2, pady=2)
            self.img_labels.append(img_label)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.static_attributes = {}
        self.total = 0

    def accept(self):
        for f, val in self.feature_inputs.items():
            self.static_attributes[f] = STATIC_ATTRIBUTES[f].index(
                val.get())
        self.total = len(self.df)
        self.top.destroy()

    def cancel(self):
        self.static_attributes = {}
        self.top.destroy()


class Main:
    def __init__(self, df, max_imgs_per_track=20):
        self.df = df
        self.track_ids = list(sorted(self.df["id"].unique()))
        self.max_imgs_per_track = max_imgs_per_track
        self.root = tk.Tk()
        try:
            self.root.tk.call("source", "~/.themes/azure/azure.tcl")
            self.root.tk.call("set_theme", "light")
        except:
            pass

        self.root.title("Image annotation")

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
        self.button_row.pack(fill=tk.X)

        # initialize status bar
        self.statusbar = ttk.Label(self.root, text="",
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        # initialize video and annotation
        self.init_annotation()
        self.track_idx = -1

        # add keybinds
        self.root.bind("<space>", lambda _: self.step_until_new_tracklet())
        self.root.bind("<a>", lambda _: self.annotate_track())
        self.root.bind("<Control-s>", lambda _: self.save_csv())
        self.root.bind("<Button-1>", lambda _: self.root.focus())
        self.root.config(menu=self.menubar)
        self.root.mainloop()

    def init_annotation(self):
        static_features = list(
            STATIC_ATTRIBUTES.keys())
        self.annotator = Annotator(static_features)

    def update_tracklet_options(self):
        options = []
        for idx in reversed(self.track_ids[max(0, self.track_idx - 10):(self.track_idx + 1)]):
            track_id = str(idx)
            options.append(track_id)
        self.tracklet_options["values"] = options
        self.tracklet_choice.set(str(self.track_ids[self.track_idx]))

    def step_until_new_tracklet(self):
        if self.track_idx < len(self.track_ids) - 1:
            self.track_idx += 1
            self.update_tracklet_options()
        else:
            self.statusbar["text"] = "Video ended."

    def annotate_track(self):
        track_id = int(self.tracklet_choice.get())
        if track_id not in self.track_ids:
            self.statusbar["text"] = "Invalid track id."
            return

        track_annot = TrackAnnotatorDialog(
            self.root, df, track_id, self.max_imgs_per_track)
        self.root.wait_window(track_annot.top)
        if len(track_annot.static_attributes):
            self.annotator.add_annotations(
                track_id, track_annot.static_attributes)
            self.statusbar["text"] = f"Track {track_id} added ({track_annot.total} total)."
        else:
            self.statusbar["text"] = f"Track {track_id} aborted."

    def delete_track(self):
        try:
            track_id = int(self.tracklet_choice.get())
            self.annotator.del_annotations(track_id)
            self.statusbar["text"] = "Track {} annotations deleted.".format(
                track_id)
        except ValueError:
            self.statusbar["text"] = "Error: deleting track unsuccessful."

    def save_csv(self):
        path = filedialog.asksaveasfilename()
        if path:
            self.annotator.save_as_csv(path)
            self.statusbar["text"] = "Annotations saved as csv."
        else:
            self.statusbar["text"] = "Invalid path chosen"


df = pd.read_csv(args.df_path)
df = df[df["dataset"] == "cityflow"]
df = df[df["subset"] == "train"]
df["id"] = df["id"].astype(int)
df = df.sample(frac=1.0)
print(f"Df loaded, rows: {len(df)}.")

main = Main(df, max_imgs_per_track=30)
