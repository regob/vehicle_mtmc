import imageio
import argparse
import shutil
import os
import sys
import tqdm

parser = argparse.ArgumentParser(
    description="Extract one or all frames from a video as images.")
parser.add_argument("video", help="video file to extract frames from")
parser.add_argument(
    "save_path", help="""Path to save image(s) to. If a single image is to be extracted, """
    """this is a file path, else a target directory""")
parser.add_argument("--frame_num", type=int, default=-1,
                    help="number of a single frame to extract (from 0). if not provided, all frames are extracted.")
args = parser.parse_args()

if args.frame_num >= 0:
    save_dir, save_file = os.path.split(args.save_path)
    if save_dir == "":
        save_dir = "."
else:
    save_dir = args.save_path

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


video = imageio.get_reader(args.video)
for frame_num, frame in tqdm.tqdm(enumerate(video), total=video.count_frames()):
    if args.frame_num >= 0:
        if args.frame_num > frame_num:
            continue
        if args.frame_num < frame_num:
            break

        imageio.imsave(args.save_path, frame)

    else:
        imageio.imsave(f"frame_{frame_num}.jpg", frame)
