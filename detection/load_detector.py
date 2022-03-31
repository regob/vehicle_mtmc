from pathlib import Path
import torch
import os


def load_yolo(which):
    """Load a yolo network from local repository. Download the weights there if needed."""

    cwd = Path.cwd()
    yolo_dir = str(Path(__file__).parent.joinpath("yolov5"))
    os.chdir(yolo_dir)
    model = torch.hub.load(yolo_dir, which, source="local")
    os.chdir(str(cwd))
    return model
