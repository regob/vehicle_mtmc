import sys
import os

from yacs.config import CfgNode

from mot.run_tracker import run_mot
from mtmc.run_mtmc import run_mtmc
from mtmc.output import save_tracklets_per_cam, save_tracklets_csv_per_cam, annotate_video_mtmc
from config.defaults import get_cfg_defaults
from config.config_tools import expand_relative_paths
from config.verify_config import check_express_config, global_checks, check_mot_config
from tools.util import parse_args
from tools import log


def run_express_mtmc(cfg: CfgNode):
    """Run Express MTMC on a given config."""
    if not check_express_config(cfg):
        sys.exit(2)
    mot_configs = []
    cam_names, cam_dirs = [], []
    for cam_idx, cam_info in enumerate(cfg.EXPRESS.CAMERAS):
        cam_cfg = cfg.clone()
        cam_cfg.defrost()
        for key, val in cam_info.items():
            setattr(cam_cfg.MOT, key.upper(), val)

        cam_video_name = os.path.split(cam_cfg.MOT.VIDEO)[1].split(".")[0]
        cam_names.append(cam_video_name)

        # set output dir of MOT to a unique folder under the root OUTPUT_DIR
        cam_dir = os.path.join(cfg.OUTPUT_DIR, f"{cam_idx}_{cam_video_name}")
        cam_dirs.append(cam_dir)
        cam_cfg.OUTPUT_DIR = cam_dir
        cam_cfg.freeze()

        mot_configs.append(cam_cfg)
        if not check_mot_config(cam_cfg):
            log.error(
                f"Error in the express config of camera {len(mot_configs) - 1}.")
            sys.exit(2)

    # run MOT in all cameras
    for mot_conf in mot_configs:
        run_mot(mot_conf)

    log.info("Express: Running MOT on all cameras finished. Running MTMC...")

    # run MTMC
    pickle_paths = [os.path.join(path, f"{cam_name}.pkl")
                    for path, cam_name in zip(cam_dirs, cam_names)]
    mtmc_cfg = cfg.clone()
    mtmc_cfg.defrost()
    mtmc_cfg.MTMC.PICKLED_TRACKLETS = pickle_paths
    mtmc_cfg.freeze()
    mtracks = run_mtmc(mtmc_cfg)

    log.info("Express: Running MTMC on all cameras finished. Saving final results ...")

    # save single cam tracks
    final_pickle_paths = [path.split(".")[0] + "_mtmc.pkl" for path in pickle_paths]
    final_csv_paths = [path.split(".")[0] + "_mtmc.csv" for path in pickle_paths]
    save_tracklets_per_cam(mtracks, final_pickle_paths)
    save_tracklets_csv_per_cam(mtracks, final_csv_paths)

    if cfg.EXPRESS.FINAL_VIDEO_OUTPUT:
        for i, pkl_path in enumerate(pickle_paths):
            video_in = mot_configs[i].MOT.VIDEO
            video_ext = video_in.split(".")[1]
            video_out = pkl_path.split(".")[0] + f"_mtmc.{video_ext}"
            annotate_video_mtmc(video_in, video_out, mtracks, i, font=cfg.MOT.FONT)


if __name__ == "__main__":
    args = parse_args("Express MTMC: run MOT on all cameras and then MTMC.")
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

    log.log_init(os.path.join(cfg.OUTPUT_DIR, args.log_filename),
                 args.log_level, not args.no_log_stdout)
    run_express_mtmc(cfg)
