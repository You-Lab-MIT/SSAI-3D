"""Surgeon: rank network layers for sparse fine-tuning.

Zero-shot proxies (weight statistics, gradient norm, SNIP, GraSP, Fisher,
plain, SynFlow) are computed for every Conv2d/Linear layer of the pre-trained
network on a few sample crops. A small pre-trained MLP maps each layer's
8 normalised proxies to a score; layers are returned best first.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from basicsr.models import create_model
from basicsr.utils import (compute_fisher_per_weight, compute_grasp_per_weight,
                           compute_plain_per_weight, compute_snip_per_weight,
                           compute_synflow_per_weight, get_grad_norm_arr,
                           img2tensor)
from ssai3d.config import load_opt

SURGEON_WEIGHTS = Path(__file__).parent / 'assets' / 'surgeon_mlp.pt'

ZERO_SHOT_METRICS = {
    'grad_norm': get_grad_norm_arr,
    'snip': compute_snip_per_weight,
    'grasp': compute_grasp_per_weight,
    'fisher': compute_fisher_per_weight,
    'plain': compute_plain_per_weight,
    'synflow': compute_synflow_per_weight,
}


def _layers(net):
    return [m for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def _minmax(values, lo, hi):
    return (values - lo) / (hi - lo)


class Surgeon:
    """Computes the per-layer zero-shot features of a pre-trained network (on CPU)."""

    def __init__(self, model_path, zs_dir, num_samples=2, config='finetune'):
        opt = load_opt(config, is_train=True, device='cpu',
                       overrides={'path': {'pretrain_network_g': str(model_path)}})
        self.net = create_model(opt).net_g
        self.zs_dir = zs_dir
        self.num_samples = num_samples
        self.criterion = nn.CrossEntropyLoss()

    def load_samples(self):
        """First ``num_samples`` zero-shot crops as unnormalised float tensors."""
        names = sorted(os.listdir(os.path.join(self.zs_dir, 'zs_lq')))[:self.num_samples]
        stacks = []
        for sub in ('zs_lq', 'zs_gt'):
            imgs = [cv2.imread(os.path.join(self.zs_dir, sub, n)) for n in names]
            stacks.append(torch.tensor(np.stack(img2tensor(imgs)).astype(np.float32)))
        return stacks

    def features(self):
        """Return an (n_layers, 8) tensor of normalised zero-shot features."""
        lq, gt = self.load_samples()

        self.criterion(self.net(lq), gt).backward()
        layers = _layers(self.net)
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = True

        lq, gt = self.load_samples()
        proxies = {name: fn(self.net, lq, gt, self.criterion)
                   for name, fn in ZERO_SHOT_METRICS.items()}

        # Read after the proxies, as in the published method: SynFlow leaves the
        # weights as |w| in float64, and the surgeon MLP was trained on these values.
        weights = [layer.weight for layer in layers]
        mean = np.array([w.mean().item() for w in weights])
        var = np.array([w.var().item() for w in weights])
        columns = [_minmax(mean, mean.min(), mean.max()),
                   _minmax(var, var.min(), var.max())]
        for values in proxies.values():
            per_layer = np.array([v.mean().item() for v in values])
            scalar = np.array([v.item() if v.numel() == 1 else v.mean().item()
                               for v in values])
            columns.append(_minmax(scalar, per_layer.min(), per_layer.max()))
        return torch.tensor(np.stack(columns, 1), dtype=torch.float32)


class SurgeonMLP(nn.Module):
    """Scores a layer from its zero-shot features (higher is better)."""

    def __init__(self, input_dims=8, weights=SURGEON_WEIGHTS):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, 64), nn.GELU(),
            nn.Linear(64, 128), nn.Dropout(0.2), nn.GELU(),
            nn.Linear(128, 128), nn.Dropout(0.2), nn.GELU(),
            nn.Linear(128, 128), nn.Dropout(0.2), nn.GELU(),
            nn.Linear(128, 64), nn.Dropout(0.2), nn.GELU(),
            nn.Linear(64, 1))
        if weights is not None:
            self.mlp.load_state_dict(torch.load(weights, weights_only=True))

    @torch.no_grad()
    def rank(self, features):
        """Return [(layer_idx, -score)] sorted best first; layers are numbered from 1."""
        self.mlp.eval()
        scores = [-self.mlp(f).item() for f in features]
        return sorted(zip(range(1, len(scores) + 1), scores), key=lambda x: x[1])


def rank_layers(model_path, zs_dir, num_samples=2):
    """Rank the layers of the network at ``model_path`` using crops in ``zs_dir``."""
    features = Surgeon(model_path, zs_dir, num_samples).features()
    return SurgeonMLP(features.shape[1]).rank(features)
