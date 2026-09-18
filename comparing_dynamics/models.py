import torch
import torch.nn as nn
from collections.abc import Iterable

'''
Implementation of PINNs. 

Source: https://github.com/AdityaLab/pinnsformer/blob/main/model/pinn.py
'''
ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}

def get_model(model_name, model_params):
  return MODEL_OPTIONS[model_name](**model_params)

class FourierEmbs(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, embed_scale: float = 1.0):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.kernel = nn.Parameter(torch.randn(input_dim, embed_dim // 2) * embed_scale)
        # this ads (input_dim, embed_dim // 2) trainable parameters

    def forward(self, x):
        proj = x @ self.kernel
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class PINN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=200, out_dim=1, num_layers=4, use_layer_normalization=False, fourier_emb: dict = None,  # e.g., {"embed_dim": 64, "embed_scale": 1.0}
    ):
        super(PINN, self).__init__()

        # Initialize Fourier embedding if provided
        self.fourier = None
        if fourier_emb is not None:
            self.fourier = FourierEmbs(input_dim=in_dim, **fourier_emb)
            in_dim = fourier_emb["embed_dim"]  # replace input dimension after Fourier embedding

        layers = []
        for i in range(num_layers - 1):
            in_features = in_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_features=in_features, out_features=hidden_dim))
            if use_layer_normalization:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x, t):
        src = torch.cat((x, t), dim=-1)  # shape: (batch, in_dim)
        if self.fourier is not None:
            src = self.fourier(src)
        return self.linear(src)

class MNISTMLP(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, num_classes=10, activation_fn="tanh", use_layer_normalization=False):
        super().__init__()

        activation_fn = ACTIVATIONS[activation_fn]

        in_dim = 28 * 28
        layers = []

        # Case 1: hidden_dim is a list/tuple of layer sizes
        # Case 2: hidden_dim is a single number
        if isinstance(hidden_dim, Iterable) and not isinstance(hidden_dim, (int, float)):
            hidden_dims = list(hidden_dim)
        else:
            hidden_dims = [hidden_dim] * num_layers

        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_layer_normalization:
                layers.append(nn.LayerNorm(h))
            layers.append(activation_fn())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

MODEL_OPTIONS = {
    "mnistMLP": MNISTMLP,
    "pinn": PINN,
}

#TODO CNN
