import argparse
import os
import sys
import pickle
import logging

from config.defaults import get_cfg_defaults
from mtmc.cameras import CameraLayout
from mtmc.mtmc_matching import greedy_mtmc_matching
from tools import log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MTMC matching with MOT results already available on all cameras.")
    parser.add_argument("--config", help="config yaml file")
    parser.add_argument("--log_level", default="info", help="logging level")
    parser.add_argument("--tee_stdout", default=True, type=bool, help="show log on stdout too")
    return parser.parse_args()


def verify_config(cfg):
    if not cfg.OUTPUT_DIR or not isinstance(cfg.OUTPUT_DIR, str):
        log.error("OUTPUT_DIR config param is mandatory.")
    elif not cfg.MTMC.CAMERA_LAYOUT:
        log.error("MTMC.CAMERA_LAYOUT param is mandatory.")
    else:
        return
    log.error("Exiting due to errors ...")
    sys.exit(2)
########################################
# Parse args and configuration
########################################

args = parse_args()
cfg = get_cfg_defaults()
if args.config:
    cfg.merge_from_file(os.path.join(cfg.SYSTEM.CFG_DIR, args.config))
cfg.freeze()
verify_config(cfg)
    
if not os.path.isabs(cfg.OUTPUT_DIR):
    OUTPUT_DIR = os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.OUTPUT_DIR)
else:
    OUTPUT_DIR = cfg.OUTPUT_DIR
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
MTMC_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "mtmc")
if not os.path.exists(MTMC_OUTPUT_DIR):
    os.makedirs(MTMC_OUTPUT_DIR)

log_level_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
log_level = log_level_map[args.log_level]
log_path = os.path.join(OUTPUT_DIR, "mtmc", "mtmc_log.txt")
log.log_init(log_path, log_level, tee_stdout=args.tee_stdout)

########################################
# Run MTMC
########################################


def load_pickle(pth: str):
    """Load a pickled tracklet file."""
    if not os.path.isabs(pth):
        pth = os.path.join(cfg.SYSTEM.ROOT_DIR, pth)
    with open(pth, "rb") as f:
        res = pickle.load(f)
    return res


# load camera layout
cams = CameraLayout(os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.MTMC.CAMERA_LAYOUT))
log.info("Camera layout loaded with %s cams.", cams.n_cams)

# load tracklets
tracks = []
if len(cfg.MTMC.PICKLED_TRACKLETS) != cams.n_cams:
    log.error("Number of pickled tracklets (%s) != number of cameras (%s)",
              len(cfg.MTMC.PICKLED_TRACKLETS), cams.n_cams)
    sys.exit(1)

for path in cfg.MTMC.PICKLED_TRACKLETS:
    tracks.append(load_pickle(path))
    log.info("Tracklets loaded for camera %s: %s in total.",
             len(tracks) - 1, len(tracks[-1]))

for cam_tracks in tracks:
    for track in cam_tracks:
        track.compute_mean_feature()

multicam_tracks = greedy_mtmc_matching(tracks, cams, linkage="single")
mtmc_pickle_path = os.path.join(OUTPUT_DIR, "mtmc", "mtmc_tracklets.pkl")
with open(mtmc_pickle_path, "wb") as f:
    pickle.dump(multicam_tracks, f, pickle.HIGHEST_PROTOCOL)
log.info("MTMC result (%s tracks) saved to: %s", len(multicam_tracks), mtmc_pickle_path)
