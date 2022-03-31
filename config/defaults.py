from yacs.config import CfgNode as CN
from pathlib import Path
import os

C = CN()
C.SYSTEM = CN()

# path to the config directory
C.SYSTEM.CFG_DIR = str(Path(__file__).parent)

# path to the repo's root directory
C.SYSTEM.ROOT_DIR = str(Path(__file__).parent.parent)

# number of GPUs to use (currently 0 or 1)
C.SYSTEM.NUM_GPUS = 1

########################################
# MOT (single camera tracking) config
########################################
C.MOT = CN()

# video input for tracking, to be overridden
C.MOT.VIDEO = "undefined"

# reid model opts file for loading
C.MOT.REID_MODEL_OPTS = "undefined"

# reid model checkpoint file
C.MOT.REID_MODEL_CKPT = "undefined"

# use half precision (fp16) instead of fp32 in reid model
C.MOT.REID_FP16 = True

# batch size for running the reid model
C.MOT.REID_BATCHSIZE = 8

# object detector (yolov5s, yolov5m, yolov5l, other yolov5 versions)
C.MOT.DETECTOR = "yolov5l"

# show video output stream for tracking (cv2 library NEEDED)
C.MOT.SHOW = True

# video stream save path (or None for not saving the video)
C.MOT.VIDEO_OUTPUT = None

# path for saving the results in a pickled format (with re-id features)
C.MOT.RESULT_PATH = None

# path for saving the results into a csv file without re-id features (or None for not saving)
C.MOT.CSV_RESULT_PATH = None

# font for text subscriptions (id labels, etc.)
C.MOT.FONT = "assets/NimbusRomNo9L-Regu.ttf"

# minimum number of bounding boxes per track
C.MOT.MIN_FRAMES = 3

########################################
# REID model training / testing config
########################################
C.REID = CN()


def get_cfg_defaults():
    """Get a yacs config object with default values."""
    return C.clone()
