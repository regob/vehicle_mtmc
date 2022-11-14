from typing import List, Union
import pickle
import torch
import numpy as np

from mot.projection_3d import Projector, dist, dist_planar
from tools import log
from tools.preprocessing import create_extractor


STATIC_ATTRIBUTES = {
    "color": ["yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black",
              "purple", "pink"],
    "type": ["sedan", "suv", "van", "hatchback", "mpv",
             "pickup", "bus", "truck", "estate", "sportscar", "RV", "bike"],
}

DYNAMIC_ATTRIBUTES = {
    "brake_signal": ["off", "on"],
}


def get_attribute_value(name: str, value: int):
    """Get the description of an attribute, e.g. get_attribute_value('color', 5) -> 'blue'."""
    if name == "speed":
        return str(value)
    if name in STATIC_ATTRIBUTES:
        return STATIC_ATTRIBUTES[name][value]
    if name in DYNAMIC_ATTRIBUTES:
        return DYNAMIC_ATTRIBUTES[name][value]
    err = f"Invalid static or dynamic attribute name: {name}."
    raise ValueError(err)


def net_is_convolutional(model: torch.nn.Module):
    if isinstance(model, torch.nn.Conv2d):
        return True

    for child in model.children():
        if net_is_convolutional(child):
            return True
    return False


class AttributeExtractor:
    """Base class for extracting dynamic and static attributes from images and re-id features."""

    def __init__(self, models):
        self.models = models
        model = next(iter(models.values()))
        self.device = next(iter(model.parameters())).device
        self.dtype = next(iter(model.parameters())).dtype

        self.attribute_idx = {k: i for i, k in enumerate(self.models.keys())}
        self.num_attributes = len(self.attribute_idx)
        self.attribute_name = {v: k for k, v in self.attribute_idx.items()}

    def __call__(self, X: torch.Tensor, batch_size=1):
        """Computes attributes from image inputs or re-id feature inputs."""
        out = self._run_extract(X, batch_size).cpu().numpy()
        result = {}
        for attrib, idx in self.attribute_idx.items():
            result[attrib] = list(out[:, idx])
        return result

    def _run_extract(self, X, batch_size):
        """Extract attributes from X using either CNN or FCNN models."""
        num_samples = X.shape[0]
        X = X.type(self.dtype)
        out = torch.zeros((num_samples, self.num_attributes), dtype=torch.int32,
                          device=self.device)

        for attrib, model in self.models.items():
            attrib_idx = self.attribute_idx[attrib]
            for i in range(0, num_samples, batch_size):
                imax = min(num_samples, i + batch_size)
                X_in = X[i:imax]
                with torch.no_grad():
                    Y = model(X_in.to(self.device))
                    out[i:imax, attrib_idx] = Y.argmax(1)
        return out.to("cpu")


class AttributeExtractorMixed:
    """Computes attributes using FCNN or/and CNN models."""

    def __init__(self, model_paths_by_attribute, fp16=False, device="cuda:0", batch_size=1):
        # torch models that run on reid embeddings / those that run on images (CNN)
        self.models_reid, self.models_img = {}, {}
        # generic models (e.g sklearn, that run on reid emeddings)
        self.models_reid_generic = {}
        self.batch_size = batch_size

        for name, path in model_paths_by_attribute.items():
            if path.endswith((".pth", ".pt")):
                model = torch.load(path)
                model.eval()
                if fp16:
                    model.half()
                model.to(device)
                if net_is_convolutional(model):
                    self.models_img[name] = model
                else:
                    self.models_reid[name] = model
            elif path.endswith(".pkl"):
                with open(path, "rb") as f:
                    model = pickle.load(f)
                self.models_reid_generic[name] = model
            else:
                log.error(f"Attribute extractor format not supported: {path}. Use .pkl, .pth or .pt")
        self.reid_extractor = None if len(
            self.models_reid) == 0 else AttributeExtractor(self.models_reid)
        if len(self.models_img) == 0:
            self.cnn_extractor = None
        else:
            self.cnn_extractor = create_extractor(
                AttributeExtractor, models=self.models_img, batch_size=batch_size)
        log.debug(f"Attribute extractors loaded. Exracted from re-id: {list(self.models_reid.keys())}, "
                  f"Extracted from images: {list(self.models_img.keys())}, "
                  f"Extracted from reid by generic models: {list(self.models_reid_generic.keys())}.")

    def __call__(self, frame: np.ndarray, bboxes: List[Union[List, np.ndarray]], X_reid: torch.Tensor):
        """Computes attributes from image inputs and/or re-id feature inputs."""
        result = {}

        # if no bounding boxes on the frame, return empty list for each attribute
        if len(bboxes) == 0:
            for attr in list(self.models_img.keys()) + list(self.models_reid.keys()) + \
                list(self.models_reid_generic.keys()):
                result[attr] = []
            return result
        
        # run prediction for generic models (sklearn) on reid embeddings
        for attr, model in self.models_reid_generic.items():
            result[attr] = list(model.predict(X_reid))

        # predict from reid embeddings using torch FCNN networks (if any)
        if self.reid_extractor is not None:
            for k, v in self.reid_extractor(X_reid, batch_size=self.batch_size).items():
                result[k] = v

        # predict from images using torch CNNs (if any)
        if self.cnn_extractor is not None:
            res = self.cnn_extractor(frame, bboxes)
            if res:
                for k, v in res.items():
                    result[k] = v
        return result


class SpeedEstimator:
    def __init__(self, projector: Projector, frame_rate):
        self.projector = projector
        self.frame_rate = frame_rate
        
    def average_speed(self, coords: list, total_frames: int, max_dist_ratio=2.0):
        """Average speed of an object over multiple frames."""
        if len(coords) < 2 or total_frames == 0:
            return 0.0
        coords = [self.projector.project3d(x, y) for x, y in coords]
        total_dist = dist(coords[0], coords[-1])
        dists = [dist_planar(coords[i], coords[i+1]) for i in range(len(coords)-1)]
        partial_dist = sum(dists)
        real_dist = partial_dist if partial_dist / total_dist <=  max_dist_ratio else total_dist
        return real_dist * (self.frame_rate / total_frames)  * 3.6

