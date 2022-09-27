<h1 align="center"> MTMC Tracking</h1>
<h3>A modular framework for Multi-target Multi-camera (MTMC) object tracking.</h3>

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/github/regob/vehicle_mtmc)](https://lgtm.com/projects/g/regob/vehicle_mtmc/context:python)
[![Total
alerts](https://img.shields.io/lgtm/alerts/github/regob/vehicle_mtmc?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/regob/vehicle_mtmc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repo contains baseline code for 
 - **Multi-Object Tracking (MOT):** Detecting (Yolov5) and tracking (DeepSORT,
   ByteTrack) objects in video streams.  
 - **Determining object attributes:** (like color, type in vehicles).  
 - **Multi-target multi-camera tracking (MTMC):** Match tracks across cameras
    after running MOT in a multi-camera system.  
 - **Evaluation:** Calculate MOT/MTMC metrics (MOTA, IDF1) automatically if
     ground truth annotations are provided.  
 - **Express run:** Run everything from above in one fly.  
  

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
