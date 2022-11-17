<h1 align="center"> MTMC Tracking</h1>
<h3>A modular framework for Multi-target Multi-camera (MTMC) object tracking.</h3>
 
<!-- [![Language grade: Python](https://img.shields.io/lgtm/grade/python/github/regob/vehicle_mtmc)](https://lgtm.com/projects/g/regob/vehicle_mtmc/context:python)
[![Total
alerts](https://img.shields.io/lgtm/alerts/github/regob/vehicle_mtmc?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/regob/vehicle_mtmc/)
-->

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repo contains baseline code for 
 - **Multi-Object Tracking (MOT):** Detecting (Yolov5) and tracking (DeepSORT,
   ByteTrack) objects in video streams.  
 - **Determining object attributes:** (like color, type in vehicles, or speed estimation if camera calibration is performed).  
 - **Multi-target multi-camera tracking (MTMC):** Match tracks across cameras
    after running MOT in a multi-camera system.  
 - **Evaluation:** Calculate MOT/MTMC metrics (MOTA, IDF1) automatically if
     ground truth annotations are provided.  
 - **Express run:** Run everything from above in one fly.  
  
## Installation

- Nvidia drivers have to be installed (check with `nvidia-smi`), preferably supporting CUDA >11.0.
- Tested on python3.7 to 3.10.
- The `requirements.txt` contains a working configuration with torch `1.12.0`, but installing the packages manually with different versions can work too.

Creating a virtual environment is **highly** recommended, except if working in a disposable environment (Kaggle, Colab, etc). Before installing `requirements.txt` cython needs
to be installed:
```bash
pip install cython numpy
```
then install the rest:
```bash
pip install -r requirements.txt
```

## Download pretrained models

Some pretrained models can be downloaded from [Google drive](https://drive.google.com/file/d/1STbsacssLtlHpUesNzuTeUPrfMlWbSKu/view). Create a `models` subdirectory, and unzip the models there. It contains:
- A resnet50-ibn re-id model trained on VeRi-Wild, CityFlow, VRIC, and some private data.
- SVM classifiers for vehicle color/type running on the re-id embeddings.

## MOT

<img alt="Highway example video" src="assets/highway_tracked.gif">

Running single-cam tracking requires at a minimum a video, a re-id model and a configuration file. A fairly minimal configuration file for the `highway.mp4` example video and pretrained re-id model is below:
```yaml
OUTPUT_DIR: "output/mot_highway"
MOT:
  VIDEO: "datasets/highway.mp4"
  REID_MODEL_OPTS: "models/resnet50_mixstyle/opts.yaml"
  REID_MODEL_CKPT: "models/resnet50_mixstyle/net_19.pth"
  DETECTOR: "yolov5x6"
  TRACKER: "bytetrack_iou"
  SHOW: false
  VIDEO_OUTPUT: true
```
If the config file is at `config/examples/mot_highway.yaml`, tracking can be run from the repo root with (PYTHONPATH needs to be set to the root folder):
```bash
$ export PYTHONPATH=$(pwd)
$ python3 mot/run_tracker.py --config examples/mot_highway.yaml
```

The required parameters for MOT are (paths can be relative to the repo root, or absolute):
- `OUTPUT_DIR`: Directory, where the outputs will be saved.
- `MOT.VIDEO`: Path to the video input.
- `MOT.REID_MODEL_OPTS`: path to the `opts.yaml` of the reid model.
- `MOT.REID_MODEL_CKPT`: path to the checkpoint of the reid model.

Other important parameters:
- `MOT.DETECTOR`: yolov5 versions are supported.
- `MOT.TRACKER`: Choose between ByteTrack ("bytetrack_iou") or DeepSORT ("deepsort").
- `MOT.SHOW`: Show tracking online in a window (cv2 needs to connect to display for this, or it crashes).
- `MOT.VIDEO_OUTPUT`: Save tracked video in the output folder.
- `MOT.STATIC_ATTRIBUTES`: Configure attribute extraction models.
- `MOT.CALIBRATION`: Camera calibration file (to be described below).

### Static attributes
Determining static attributes (e.g. type, color) can be configured as:
```yaml
MOT:
  STATIC_ATTRIBUTES:
    - color: "models/color_svm.pkl"
    - type: "models/type_svm.pkl"
```
Models can be the following:
- pytorch CNN, that gets the image in the bounding box as input
- pytorch fully-connected NN that predicts the attribute from the re-id embedding.
- sklearn/xgboost/etc models that are pickled, and have a `predict(x)` method that predicts from the re-id embedding as a numpy array.

### Camera calibration and speed estimation
Camera calibration has to be performed with the [Cal_PNP](https://github.com/zhengthomastang/Cal_PnP) package to get a homography matrix, then the path to the homography matrix has to be configured in `MOT.CALIBRATION`. An example homography matrix file is provided for `highway.mp4` at [config/examples/highway_calibration.txt](config/examples/highway_calibration.txt).

## Express MTMC

Express Multi-camera tracking runs MOT on all cameras and then hierarchical clustering on single-camera tracks. Temporal constraints are also considered, and have to be pre-configured in the `MTMC.CAMERA_LAYOUT` parameter. An example config for CityFlow S02 (4 cameras at a crossroad) is at [config/cityflow/express_s02.yaml](config/cityflow/express_s02.yaml). Its part describing the MTMC config is:
```yaml
MTMC:
  CAMERA_LAYOUT: 'config/cityflow/s02_camera_layout.txt'
  LINKAGE: 'average'
  MIN_SIM: 0.5
EXPRESS:
  FINAL_VIDEO_OUTPUT: true
  CAMERAS:
    - "video": "datasets/cityflow_track3/validation/S02/c006/vdo.avi"
      "detection_mask": "assets/cityflow/c006_mask.jpg"
      "calibration": "datasets/cityflow_track3/validation/S02/c006/calibration.txt"
    - "video": "datasets/cityflow_track3/validation/S02/c007/vdo.avi"
      "detection_mask": "assets/cityflow/c007_mask.jpg"
      "calibration": "datasets/cityflow_track3/validation/S02/c007/calibration.txt"
    - "video": "datasets/cityflow_track3/validation/S02/c008/vdo.avi"
      "detection_mask": "assets/cityflow/c008_mask.jpg"
      "calibration": "datasets/cityflow_track3/validation/S02/c008/calibration.txt"
    - "video": "datasets/cityflow_track3/validation/S02/c009/vdo.avi"
      "detection_mask": "assets/cityflow/c009_mask.jpg"
      "calibration": "datasets/cityflow_track3/validation/S02/c009/calibration.txt"
```
The MOT config is the same for all cameras, but for each camera, at least the "video" key has to be given in `EXPRESS.CAMERAS`, the meaning of the keys is the same as in the MOT config.
In the MTMC config there are only a few paramteres:
- `MTMC.LINKAGE` chooses the linkage for hierarchical clustering from ['single', 'complete', 'average'].
- `MTMC.MIN_SIM` is the minimal similarity between multi-cam tracks above which they can be merged.
- `MTMC.CAMERA_LAYOUT` stores the **mandatory** camera constraints file. The camera layout file for CityFlow S02 is at [config/cityflow/s02_camera_layout.txt](config/cityflow/s02_camera_layout.txt).
On Cityflow S02 express MTMC can be run by:
```bash
$ export PYTHONPATH=$(pwd)
$ python3 mtmc/run_express_mtmc.py --config cityflow/express_s02.yaml
```
## Finetuning/training re-id models
Models trained by my [reid/vehicle_reid](https://github.com/regob/vehicle_reid) repo are supported out-of-the-box in the configuration. Other torch models could be integrated by modifying the model loading in `mot/run_tracker.py`, which currently looks like this:
```python
# initialize reid model
reid_model = load_model_from_opts(cfg.MOT.REID_MODEL_OPTS,
                                  ckpt=cfg.MOT.REID_MODEL_CKPT,
                                  remove_classifier=True)
```

##  Acknowledgements
Some parts are adapted from other repositories:  
    - [nwojke/deep_sort](https://github.com/nwojke/deep_sort): Original
      DeepSORT code.  
    - [theAIGuysCode/yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort):
      Enhanced version of DeepSORT.   
    - [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack): Original
      ByteTrack tracker code.  

The [yolov5](https://github.com/ultralytics/yolov5) and
[vehicle_reid](https://github.com/regob/vehicle_reid) repos are used as submodules.
