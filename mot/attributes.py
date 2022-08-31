import torch


STATIC_ATTRIBUTES = {
    "color": ["yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black",
              "purple", "pink"],
    "type": ["sedan", "suv", "van", "hatchback", "mpv",
             "pickup", "bus", "truck", "estate", "sportscar", "RV", "bike"],
}

DYNAMIC_ATTRIBUTES = {
    "brake_signal": ["off", "on"],
}


def net_is_convolutional(model: torch.nn.Module):
    if isinstance(model, torch.nn.Conv2d):
        return True

    for child in model.children():
        if net_is_convolutional(child):
            return True
    return False


class AttributeExtractor:
    """This allows to extract dynamic and static attributes from images and re-id features."""

    def __init__(self, model_paths_by_attribute, fp16=False, device="cuda:0"):
        self.models_reid, self.models_img = {}, {}
        self.attribute_idx = {}

        for name, path in model_paths_by_attribute.items():
            model = torch.load(path)
            model.eval()
            if fp16:
                model.half()
            model.to(device)
            if net_is_convolutional(model):
                self.models_reid[name] = model
            else:
                self.models_img[name] = model
            self.attribute_idx[name] = len(self.attribute_idx)
        self.device = device
        self.dtype = next(iter(model.parameters())).dtype
        self.num_attributes = len(model_paths_by_attribute)
        self.attribute_name = {v: k for k, v in self.attribute_idx.items()}

    def __call__(self, X: torch.Tensor, X_reid: torch.Tensor, batch_size=16):
        """Computes attributes from image inputs and/or re-id feature inputs."""

        num_samples = X.shape[0] if X else X_reid.shape[0]
        out = torch.zeros((num_samples, self.num_attributes), dtype=torch.int32,
                          device=self.device)

        def run_extract(X, models):
            """Extract attributes from X using either CNN or FCNN models."""
            for attrib, model in models.items():
                attrib_idx = self.attribute_idx[attrib]
                for i in range(0, num_samples, batch_size):
                    imax = min(num_samples, i + batch_size)
                    X_in = X[i:imax]
                    with torch.no_grad():
                        Y = model(X_in.to(self.device))
                    out[i:imax, attrib_idx] = Y.argmax(1)

        if self.models_reid:
            X_reid = X_reid.type(self.dtype)
            run_extract(X_reid, self.models_reid)

        if self.models_img:
            X = X.type(self.dtype)
            run_extract(X, self.models_img)

        result = {}
        for attrib, idx in self.attribute_idx.items():
            result[attrib] = list(out[:, idx].cpu().numpy())
        return result
