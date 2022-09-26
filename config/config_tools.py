import os
from yacs.config import CfgNode as CN

def get_abspath(path: str, project_root: str):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(project_root, path))

def expand_relative_paths(root_cfg: CN):
    c = root_cfg
    root = c.SYSTEM.ROOT_DIR
    c.OUTPUT_DIR = get_abspath(c.OUTPUT_DIR, root)
    c.FONT = get_abspath(c.FONT, root)
    c.MOT.VIDEO = get_abspath(c.MOT.VIDEO, root)
    c.MOT.REID_MODEL_CKPT = get_abspath(c.MOT.REID_MODEL_CKPT, root)
    c.MOT.REID_MODEL_OPTS = get_abspath(c.MOT.REID_MODEL_OPTS, root)
    c.MOT.DETECTION_MASK = get_abspath(c.MOT.DETECTION_MASK, root)

    # expand paths in attribute dicts
    for x in c.MOT.STATIC_ATTRIBUTES:
        for k, v in x.items():
            x[k] = get_abspath(v, root)
    for x in c.MOT.DYNAMIC_ATTRIBUTES:
        for k, v in x.items():
            x[k] = get_abspath(v, root)
    c.MOT.ZONE_MASK_DIR = get_abspath(c.MOT.ZONE_MASK_DIR, root)

    c.MTMC.CAMERA_LAYOUT = get_abspath(c.MTMC.CAMERA_LAYOUT, root)
    for i, x in enumerate(c.MTMC.PICKLED_TRACKLETS):
        c.MTMC.PICKLED_TRACKLETS[i] = get_abspath(x, root)

    # expand paths in express config
    for it in c.EXPRESS.CAMERAS:
        for k, v in list(it.items()):
            if k == "valid_zonepaths":
                continue
            it[k] = get_abspath(v, root)

    # expand paths in evaluation config
    c.EVAL.GROUND_TRUTHS = list(map(lambda x: get_abspath(x, root), c.EVAL.GROUND_TRUTHS))
    c.EVAL.PREDICTIONS = list(map(lambda x: get_abspath(x, root), c.EVAL.PREDICTIONS))
    return c
