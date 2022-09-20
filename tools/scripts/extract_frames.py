import imageio
from PIL import Image
import argparse
import os
import sys
import tqdm
import pandas as pd

from config.defaults import get_cfg_defaults
from tools.preprocessing import extract_image_patch

parser = argparse.ArgumentParser(
    description="Extract frames (or bounding boxes in frames) from a video as images.")
parser.add_argument("video", help="video file to extract frames from")
parser.add_argument(
    "save_dir", help="""Directory to save images in.""")
parser.add_argument("--frame_num", type=int, default=-1,
                    help="Index of a single frame to extract (from 0). if not provided, all frames are extracted.")
parser.add_argument("--annot_csv", default=None,
                    help="annotations for the objects to extract from the frames")
parser.add_argument("--annot_output_csv", default=None,
                    help="output path for the new annotations (with filepaths) as csv")
parser.add_argument("--resize", default=224, type=int,
                    help="resize image to this size, if <= 0, original size is kept")
parser.add_argument("--save_frames_too", action="store_true",
                    help="save the whole frame(s). Only matters when annot_csv is provided (and frames are not saved by default)")
########################################
# Parse args and configuration
########################################

args = parser.parse_args()
cfg = get_cfg_defaults()
cfg.freeze()

save_dir = args.save_dir

if not os.path.isdir(save_dir):
    if os.path.isfile(save_dir):
        raise ValueError(f"{save_dir} is a file not a directory.")
    ans = ""
    while ans.lower() not in ['y', 'n']:
        ans = input(
            f"Directory ({save_dir}) does not exist. Do u want to create it? (y/n) ")
    if ans.lower() == 'n':
        sys.exit(1)
    os.makedirs(save_dir)

if args.annot_csv is not None:
    df = pd.read_csv(args.annot_csv)
    annots = {}
    for idx, row in df.iterrows():
        frame_annots = annots.setdefault(row["frame"], [])
        frame_annots.append(idx)
else:
    df = None


########################################
# extract frames from video
########################################

_, video_name = os.path.split(args.video)
video_name = video_name.split(".")[0]
video = imageio.get_reader(args.video)
for frame_num, frame in tqdm.tqdm(enumerate(video), total=video.count_frames()):
    if args.frame_num >= 0:
        if args.frame_num > frame_num:
            continue
        if args.frame_num < frame_num:
            break

    if df is None or args.save_frames_too:
        imageio.imsave(os.path.join(
            args.save_dir, f"frame_{frame_num}.jpg"), frame)

    if df is None:
        continue

    for idx in annots.get(frame_num, []):
        row = df.iloc[idx]
        bbox = [row["bbox_topleft_x"], row["bbox_topleft_y"],
                row["bbox_width"], row["bbox_height"]]
        img = extract_image_patch(frame, bbox)
        if args.resize > 0:
            img = Image.fromarray(img)
            img = img.resize((args.resize, args.resize))
        imageio.imsave(os.path.join(args.save_dir, f"{row['track_id']}_frame_{frame_num}_{video_name}.jpg"),
                       img)


if args.annot_output_csv is not None:
    path_prefix = os.path.relpath(os.path.realpath(args.save_dir),
                                  start=os.path.join(os.path.realpath(cfg.SYSTEM.ROOT_DIR),
                                                     "datasets"))

    file_paths = []
    for _, row in df.iterrows():
        file_paths.append(os.path.join(
            path_prefix, f"{row['track_id']}_frame_{row['frame']}_{video_name}.jpg"))

    df["path"] = file_paths
    df.to_csv(args.annot_output_csv, index=False)
