import os
from typing import List, Dict, Union, Tuple
from yacs.config import CfgNode as CN
from tools import log


def _is(_type):
    def check_type(x):
        return isinstance(x, _type)
    return check_type


system_checks = {
    "CFG_DIR": os.path.isdir,
    "ROOT_DIR": os.path.isdir,
    "GPU_IDS": lambda x: len(x) > 0,
}

global_checks = {
    "OUTPUT_DIR": lambda x: os.path.isdir(x) or os.path.isdir(os.path.dirname(os.path.normpath(x))),
    "DEBUG_RUN": _is(bool),
    "FONT": lambda x: _is(str)(x) and os.path.exists(x),
    "FONTSIZE": _is(int),
}

common_mot_checks = {
    "REID_MODEL_OPTS": lambda x: os.path.isfile(x) and (x.endswith(".yaml") or x.endswith(".yml")),
    "REID_MODEL_CKPT": os.path.isfile,
    "REID_FP16": _is(bool),
    "REID_BATCHSIZE": lambda x: _is(int)(x) and x >= 1,
    "DETECTOR": lambda x: x.startswith("yolov5"),
    "TRACKER": lambda x: x in ["deepsort", "bytetrack_iou"],
    "SHOW": _is(bool),
    "ONLINE_VIDEO_OUTPUT": _is(bool),
    "VIDEO_OUTPUT": _is(bool),
    "MIN_FRAMES": lambda x: _is(int)(x) and x >= 1,
    "STATIC_ATTRIBUTES": lambda x: _is(list)(x) and all(os.path.exists(y) for z in x for y in z.values()),
    "DYNAMIC_ATTRIBUTES": lambda x: _is(list)(x) and all(os.path.exists(y) for z in x for y in z.values()),
    "ATTRIBUTE_INFER_BATCHSIZE": lambda x: _is(int)(x) and x >= 1,
    "REFINE": _is(bool),
}

isolated_mot_checks = {
    "VIDEO": os.path.isfile,
    "DETECTION_MASK": lambda x: x is None or os.path.isfile(x),
    "VALID_ZONEPATHS": lambda x: _is(list)(x) and all(_is(str)(y) for y in x),
    "ZONE_MASK_DIR": lambda x: x is None or os.path.isdir(x),
    "CALIBRATION": lambda x: x is None or os.path.isfile(x),
}

common_mtmc_checks = {
    "CAMERA_LAYOUT": lambda x: x is None or (_is(str)(x) and os.path.isfile(x)),
    "LINKAGE": lambda x: _is(str)(x) and x in ["average", "single", "complete", "mean_feature"],
    "MIN_SIM": lambda x: _is(float)(x) and x >= -1.0,
}

isolated_mtmc_checks = {
    "PICKLED_TRACKLETS": _is(list),
}

express_checks = {
    "CAMERAS": lambda x: _is(list)(x) and all(_check_express_camera(y) for y in x),
    "FINAL_VIDEO_OUTPUT": _is(bool),
}

evaluation_checks = {
    "GROUND_TRUTHS": lambda x: all(os.path.exists(y) for y in x),
    "PREDICTIONS": lambda x: all(os.path.exists(y) for y in x),
    "MIN_IOU": lambda x: _is(float)(x) and x <= 1.0,
    "IGNORE_FP": _is(bool),
    "DROP_SINGLE_CAM": _is(bool),
}


def _check_express_camera(d: dict):
    if "video" not in d:
        return False
    all_keys = ["video", "detection_mask", "zone_mask_dir", "valid_zonepaths", "calibration"]
    for k, v in d.items():
        if k not in all_keys:
            return False
        check_fn = isolated_mot_checks[k.upper()]
        if not check_fn(v):
            return False
    return True


def run_checks(checks: dict, cfg: CN):
    failed = False
    for check_name, check_fn in checks.items():
        if not check_fn(getattr(cfg, check_name)):
            failed = True
            log.error(f"Config check failed: {check_name}.")
    return not failed


def run_list_of_checks(checks: List[Tuple[Union[Dict, CN]]]):
    success = all(run_checks(check, cfg) for check, cfg in checks)
    if success:
        log.info("All config checks passed.")
    else:
        log.error("Config had errors. Aborting ...")
    return success


def check_mot_config(cfg: CN):
    """Check a MOT config for errors."""
    return run_list_of_checks([(system_checks, cfg.SYSTEM),
                               (global_checks, cfg),
                               (isolated_mot_checks, cfg.MOT),
                               (common_mot_checks, cfg.MOT)])


def check_mtmc_config(cfg: CN):
    """Check an MTMC config for errors."""
    return run_list_of_checks([(system_checks, cfg.SYSTEM),
                               (global_checks, cfg),
                               (isolated_mtmc_checks, cfg.MTMC),
                               (common_mtmc_checks, cfg.MTMC)])


def check_express_config(cfg: CN):
    """Check an Express config for errors."""
    return run_list_of_checks([(system_checks, cfg.SYSTEM),
                               (global_checks, cfg),
                               (common_mot_checks, cfg.MOT),
                               (common_mtmc_checks, cfg.MTMC),
                               (express_checks, cfg.EXPRESS)])


def check_eval_config(cfg: CN):
    """Check an evaluation config for errors."""
    return run_list_of_checks([(system_checks, cfg.SYSTEM),
                               (global_checks, cfg),
                               (evaluation_checks, cfg.EVAL)])
