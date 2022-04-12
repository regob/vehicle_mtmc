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
C.MOT.VIDEO = None

# reid model opts file for loading
C.MOT.REID_MODEL_OPTS = None

# reid model checkpoint file
C.MOT.REID_MODEL_CKPT = None

# use half precision (fp16) instead of fp32 in reid model
C.MOT.REID_FP16 = True

# batch size for running the reid model
C.MOT.REID_BATCHSIZE = 8

# object detector (yolov5s, yolov5m, yolov5l, other yolov5 versions)
C.MOT.DETECTOR = "yolov5l"

# path to the detection mask image showing the ROI (region of interest)
# in the image white pixels are included, while others (black ones) are excluded
C.MOT.DETECTION_MASK = None

# show video output stream for tracking (cv2 library NEEDED)
C.MOT.SHOW = True

# video stream save path (or None for not saving the video)
C.MOT.ONLINE_VIDEO_OUTPUT = None

# final video output (with tracklet refinement)
C.MOT.VIDEO_OUTPUT = None

# path for saving the results in a pickled format (with re-id features)
C.MOT.RESULT_PATH = None

# path for saving the results into a csv file without re-id features (or None for not saving)
C.MOT.CSV_RESULT_PATH = None

# font for text subscriptions (id labels, etc.)
C.MOT.FONT = "Hack-Regular.ttf"

# minimum number of bounding boxes per track
C.MOT.MIN_FRAMES = 5

# list of feature_name:model_path dicts of static feature extracting models
# e.g C.MOT.STATIC_FEATURES = [{"color": "color_extracting_model.pt"}]
C.MOT.STATIC_FEATURES = []

# batch_size for static feature inference
C.MOT.STATIC_FEATURE_BATCHSIZE = 8

# regular expressions describing valid paths of zones for tracks
# e.g: If only tracks that start and end in (zone 1 and 2) or (3 and 4) are good: ["1,.*,2", "3,.*,4"]
# zone strings consist of numbers separated by commas
# Zone 0 is reserved for detections that are not in any zone!
C.MOT.VALID_ZONEPATHS = []

# path to directory containing zone masks (if not provided, no zones will be used)
C.MOT.ZONE_MASK_DIR = None

# run tracklet refinement at the end (post processing), zones are needed for this
C.MOT.REFINE = True

########################################
# REID model training / testing config
########################################
C.REID = CN()


def get_cfg_defaults():
    """Get a yacs config object with default values."""
    return C.clone()
