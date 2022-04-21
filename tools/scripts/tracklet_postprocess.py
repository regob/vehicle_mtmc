import argparse
import os

from mot.tracklet_processing import load_tracklets, save_tracklets, save_tracklets_csv, refine_tracklets
from mot.zones import ZoneMatcher
from mot.video_output import annotate_video_with_tracklets
from config.defaults import get_cfg_defaults


parser = argparse.ArgumentParser(
    description="Run tracklet refinement and save to video optionally")
parser.add_argument("tracklet_pkl", help="pickled tracklets")
parser.add_argument("--config", required=True,
                    help="config yaml file, containing zone info")
parser.add_argument("--refine", action="store_true",
                    help="run refinement on tracklets")
parser.add_argument("--pkl_save_path", default=None,
                    help="file to store results in")
parser.add_argument("--csv_save_path", default=None,
                    help="path to save csv result to")
parser.add_argument("--video_in", default=None,
                    help="input video if video output is needed")
parser.add_argument("--video_out", default=None,
                    help="path to save video output to, video_in also needed.")

args = parser.parse_args()
cfg = get_cfg_defaults()
cfg.merge_from_file(os.path.join(cfg.SYSTEM.CFG_DIR, args.config))
cfg.freeze()

tracklets = load_tracklets(args.tracklet_pkl)

for tracklet in tracklets:
    tracklet.predict_final_static_features()


if args.refine:
    zone_matcher = ZoneMatcher(os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.MOT.ZONE_MASK_DIR),
                               cfg.MOT.VALID_ZONEPATHS)
    tracklets = refine_tracklets(tracklets, zone_matcher)[0]

if args.video_in and args.video_out:
    annotate_video_with_tracklets(args.video_in, args.video_out, tracklets)


if args.pkl_save_path:
    save_tracklets(tracklets, args.pkl_save_path)

if args.csv_save_path:
    save_tracklets_csv(tracklets, args.csv_save_path)
