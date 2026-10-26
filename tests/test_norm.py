"""Input scaling: v1 (per slice) is unchanged; whole-volume scaling and the
checkpoint-recorded setting used by SSAI-3D-v2 and new fine-tuning."""
import io

import numpy as np
import tifffile
import torch

from basicsr.utils import imfrombytes
from ssai3d.restore import input_norm


def _tiff_bytes(arr):
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    return buf.getvalue()


def test_slice_scaling_unchanged():
    a = np.random.default_rng(0).integers(0, 40000, (16, 20)).astype(np.uint16)
    img = imfrombytes(_tiff_bytes(a), float32=True)
    assert np.isclose(img.max(), 1.0)  # each slice scaled to its own maximum, as in v1


def test_volume_scaling_keeps_intensity():
    a = np.random.default_rng(0).integers(0, 40000, (16, 20)).astype(np.uint16)
    img = imfrombytes(_tiff_bytes(a), float32=True, norm='volume')
    np.testing.assert_allclose(img[..., 0], (a >> 8) / 255.0, rtol=1e-6)


def test_input_norm_from_checkpoint(tmp_path):
    new, old = tmp_path / 'v2.pth', tmp_path / 'v1.pth'
    torch.save({'params': {}, 'ssai3d': {'input_norm': 'volume'}}, new)
    torch.save({'params': {}}, old)
    assert input_norm(new) == 'volume'
    assert input_norm(old) == 'slice'
