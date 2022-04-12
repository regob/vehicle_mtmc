import argparse
import os

from mot.video_output import annotate_video_with_tracklets
from mot.tracklet_processing import load_tracklets, refine_tracklets
from mot.zones import ZoneMatcher
from config.defaults import get_cfg_defaults


parser = argparse.ArgumentParser(
    description="Load tracklets and a video and annotate only invalid tracks in the frames")
parser.add_argument("tracklets_pkl", help="trackets in a pickled format")
parser.add_argument("video_in", help="input video path")
parser.add_argument("video_out", help="output video path")
parser.add_argument("config", help="config file, containing MOT settings")
args = parser.parse_args()

cfg = get_cfg_defaults()
cfg.merge_from_file(os.path.join(cfg.SYSTEM.CFG_DIR, args.config))
cfg.freeze()

tracklets = load_tracklets(args.tracklets_pkl)
zone_matcher = ZoneMatcher(os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.MOT.ZONE_MASK_DIR),
                           cfg.MOT.VALID_ZONEPATHS)
invalid_tracklets = refine_tracklets(tracklets, zone_matcher)[1]

annotate_video_with_tracklets(args.video_in, args.video_out, invalid_tracklets)
