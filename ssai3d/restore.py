"""Slice-wise restoration with a trained network."""
import logging
import os

import numpy as np
import tifffile
import torch
from tqdm import tqdm

from basicsr.models import create_model
from basicsr.utils import imfrombytes, img2tensor, imwrite, tensor2img
from ssai3d.config import load_opt

logger = logging.getLogger('ssai3d')

# Tile size used when a whole slice does not fit in GPU memory.
FALLBACK_TILE = 768


def load_network(model_pth, denoise=False, device=None):
    """Load a restoration (or denoising) network in eval mode."""
    opt = load_opt('denoise' if denoise else 'finetune', is_train=False, device=device,
                   overrides={'path': {'pretrain_network_g': str(model_pth)}})
    net = create_model(opt).net_g
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


def _starts(length, tile, overlap):
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile, step))
    return starts + [length - tile]


def _ramp(length, start, end, total, overlap):
    """1D blending weights: linear ramps over the overlap on interior edges."""
    w = torch.ones(length)
    n = min(overlap, length)
    ramp = torch.linspace(1 / (n + 1), n / (n + 1), n)
    if start > 0:
        w[:n] = ramp
    if end < total:
        w[-n:] = torch.minimum(w[-n:], ramp.flip(0))
    return w


@torch.no_grad()
def run_network(net, x, tile=None, overlap=128):
    """Apply ``net`` to a (1, C, H, W) tensor, optionally in overlapping tiles.

    Overlapping tiles are blended with linear ramps. Tiled results differ slightly
    from whole-slice inference because NAFNet's local pooling depends on input size.
    """
    if tile is None:
        return net(x).cpu()
    _, c, h, w = x.shape
    out = torch.zeros((1, c, h, w))
    weight = torch.zeros((1, 1, h, w))
    for i in _starts(h, tile, overlap):
        for j in _starts(w, tile, overlap):
            patch = x[:, :, i:i + tile, j:j + tile]
            th, tw = patch.shape[2:]
            wt = torch.outer(_ramp(th, i, i + th, h, overlap), _ramp(tw, j, j + tw, w, overlap))
            out[:, :, i:i + th, j:j + tw] += net(patch).cpu() * wt
            weight[:, :, i:i + th, j:j + tw] += wt
    return out / weight


def restore(input_pth, output_pth, model_pth, denoise=False, tile=None, device=None):
    """Restore every slice in ``input_pth`` and write it to ``output_pth``.

    Inputs are read exactly as in the published pipeline (8-bit, scaled to the
    slice maximum). Outputs keep the input file names. Restored slices are
    saved as the channel mean of the network's uint16 output (single-channel
    float32). Denoised slices are saved as 3-channel uint16, as in v1.0,
    because they are read again as network input.

    Args:
        tile (int | None): Process slices in tiles of this size. With None,
            whole slices are used and tiling starts automatically on an
            out-of-memory error.
    """
    os.makedirs(output_pth, exist_ok=True)
    net = load_network(model_pth, denoise, device)
    device = next(net.parameters()).device
    names = sorted(os.listdir(input_pth))
    for name in tqdm(names, desc=os.path.basename(os.path.normpath(input_pth))):
        with open(os.path.join(input_pth, name), 'rb') as f:
            img = imfrombytes(f.read(), float32=True)
        x = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
        try:
            out = run_network(net, x, tile)
        except torch.cuda.OutOfMemoryError:
            if tile is not None:
                raise
            torch.cuda.empty_cache()
            tile = FALLBACK_TILE
            logger.warning(f'Out of GPU memory on {name}; switching to {tile}px tiles.')
            out = run_network(net, x, tile)
        result = tensor2img(out)
        if denoise:
            imwrite(result, os.path.join(output_pth, name))
            continue
        if result.ndim == 3:
            result = result.mean(-1)
        tifffile.imwrite(os.path.join(output_pth, name), result.astype(np.float32))
