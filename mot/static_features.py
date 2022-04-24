import torch
import numpy as np


FEATURES = {
    "color": ["yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black",
              "purple", "pink"],
    "type": ["sedan", "suv", "van", "hatchback", "mpv",
             "pickup", "bus", "truck", "estate", "sportscar", "RV", "bike"],
}


class StaticFeatureExtractor:
    def __init__(self, model_paths_by_feature, fp16=True, device="cuda:0"):
        self.models = {}
        for feature, path in model_paths_by_feature.items():
            model = torch.load(path)
            model.eval()
            if fp16:
                model.half()
            model.to(device)
            self.models[feature] = model
        self.device = device
        self.dtype = next(iter(model.parameters())).dtype
        self.num_features = len(self.models)

    def __call__(self, X, batch_size=16):
        X = X.type(self.dtype)
        out = np.zeros((len(X), self.num_features), dtype=np.uint32)

        for feature_idx, (feature, model) in enumerate(self.models.items()):
            for i in range(0, len(X), batch_size):
                imax = min(len(X), i + batch_size)
                X_in = X[i:imax]
                with torch.no_grad():
                    Y = model(X_in.to(self.device))
                out[i:imax, feature_idx] = Y.argmax(1).cpu().numpy()

        result = []
        for i in range(out.shape[0]):
            res_f = {}
            for feature_idx, (feature, _) in enumerate(self.models.items()):
                res_f[feature] = out[i, feature_idx]
            result.append(res_f)

        return result
