import numpy as np
import torch
from tools.preprocessing import fliplr


class FeatureExtractor:
    def __init__(self, model, feature_dim="infer"):
        self.model = model
        model.eval()
        self.device = next(iter(model.parameters())).device
        self.dtype = next(iter(model.parameters())).dtype
        self.feature_dim = feature_dim

    def __call__(self, X, batch_size=32):
        X = X.type(self.dtype)
        if self.feature_dim == "infer":
            dummy = X[0].unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(dummy)
            self.feature_dim = output.shape[1]

        out = np.zeros((len(X), self.feature_dim), np.float32)

        for i in range(0, len(X), batch_size):
            imax = min(len(X), i + batch_size)
            X_in = X[i:imax]
            X_in_flip = fliplr(X_in)
            with torch.no_grad():
                Y = self.model(X_in.to(self.device))
                Y_flip = self.model(X_in_flip.to(self.device))
            Y += Y_flip
            Y_norm = torch.norm(Y, p=2, dim=1, keepdim=True)
            Y = Y.div(Y_norm.expand_as(Y)).to("cpu")
            out[i:imax, :] = Y
        return out
