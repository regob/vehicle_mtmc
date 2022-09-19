import os
import sys
import pickle

from yacs.config import CfgNode as CN

from config.defaults import get_cfg_defaults
from config.verify_config import check_mtmc_config, global_checks
from config.config_tools import expand_relative_paths
from mtmc.cameras import CameraLayout
from mtmc.mtmc_matching import greedy_mtmc_matching
from tools import log
from tools.util import parse_args


########################################
# Run MTMC
########################################


def load_pickle(pth: str):
    """Load a pickled tracklet file."""
    with open(pth, "rb") as f:
        res = pickle.load(f)
    return res


def run_mtmc(cfg: CN):
    # check and verify config (has to be done after logging init to see errors)
    if not check_mtmc_config(cfg):
        sys.exit(2)

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
        
    # load camera layout
    cams = CameraLayout(cfg.MTMC.CAMERA_LAYOUT)
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
    mtmc_pickle_path = os.path.join(cfg.OUTPUT_DIR, "mtmc_tracklets.pkl")
    with open(mtmc_pickle_path, "wb") as f:
        pickle.dump(multicam_tracks, f, pickle.HIGHEST_PROTOCOL)
    log.info("MTMC result (%s tracks) saved to: %s",
             len(multicam_tracks), mtmc_pickle_path)

if __name__ == "__main__":
    args = parse_args(
        "Run MTMC matching with MOT results already available on all cameras.")
    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(os.path.join(cfg.SYSTEM.CFG_DIR, args.config))
    cfg = expand_relative_paths(cfg)
    cfg.freeze()

    # initialize output directory and logging
    if not global_checks["OUTPUT_DIR"](cfg.OUTPUT_DIR):
        log.error(
            "Invalid param value in: OUTPUT_DIR. Provide an absolute path to a directory, whose parent exists.")
        sys.exit(2)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    log_path = os.path.join(cfg.OUTPUT_DIR, args.log_filename)
    log.log_init(log_path, args.log_level, not args.no_log_stdout)

    run_mtmc(cfg)
