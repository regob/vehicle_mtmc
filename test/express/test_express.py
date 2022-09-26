import os
import tempfile

from config.defaults import get_cfg_defaults
from mot.run_tracker import MOT_OUTPUT_NAME
from mtmc.run_express_mtmc import run_express_mtmc, MTMC_OUTPUT_NAME
from mtmc.run_mtmc import MTMC_TRACKLETS_NAME
from tools import log


CITYFLOW_EXPRESS = "cityflow/express.yaml"


def check_mot_folder(path):
    assert os.path.isdir(path)
    for x in ["txt", "csv", "pkl"]:
        assert os.path.isfile(os.path.join(path, f"{MOT_OUTPUT_NAME}.{x}"))
        assert os.path.isfile(os.path.join(path, f"{MTMC_OUTPUT_NAME}.{x}"))


def check_mtmc_folder(path):
    assert os.path.isdir(path)
    assert os.path.isfile(os.path.join(path, f"{MTMC_TRACKLETS_NAME}.pkl"))


def test_cityflow_express():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(cfg.SYSTEM.CFG_DIR, CITYFLOW_EXPRESS))

    out_dir = tempfile.TemporaryDirectory()
    cfg.OUTPUT_DIR = out_dir.name
    cfg.MOT.SHOW = False
    cfg.DEBUG_RUN = True
    cfg.EXPRESS.FINAL_VIDEO_OUTPUT = False
    cfg.freeze()

    res = run_express_mtmc(cfg)
    assert res is not None

    # no errors should have happened
    assert log.num_errors == 0

    for cam_idx, cam_info in enumerate(cfg.EXPRESS.CAMERAS):
        cam_video_name = os.path.split(cam_info["video"])[1].split(".")[0]
        cam_dir = os.path.join(cfg.OUTPUT_DIR, f"{cam_idx}_{cam_video_name}")
        check_mot_folder(cam_dir)
    check_mtmc_folder(cfg.OUTPUT_DIR)
    out_dir.cleanup()
