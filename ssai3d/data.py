"""Data preparation for SSAI-3D.

A raw stack is a single 3D TIFF of shape (Z, Y, X) with coarse sampling along Z.
`downsample_rate` (``dr``) is the axial/lateral sampling ratio: axial slices are
upsampled ``dr`` times along Z before restoration.

Directory layout produced under a dataset folder::

    gt/, lq/          semi-synthetic training pairs (lateral slices, blurred + subsampled)
    zs_gt/, zs_lq/    small crops used by the surgeon to rank layers
    test_xz/          axial slices raw[:, :, x], upsampled to (Z*dr, Y)
    test_yz/          axial slices raw[:, y, :], upsampled to (Z*dr, X)
    test_xy/          lateral slices raw[z] (only used for denoising)
"""
import os
import random

import cv2
import numpy as np
import tifffile
from scipy import signal
from skimage import exposure
from tqdm import tqdm

__all__ = [
    'normalize', 'resize', 'gaussian_kernels', 'semi_synthetic_creation',
    'denoised_semi_synthetic_creation', 'generate_raw_data',
    'generate_zs_dataset', 'read_slice', 'fuse_views', 'restore_volume',
    'adjust_contrast'
]


def normalize(data):
    """Scale to the full uint16 range (truncating, as in the original code)."""
    data = data / data.max()
    data = data * 65535
    return data.astype(np.uint16)


def resize(img, dr, orig):
    """Keep every ``dr``-th row, then interpolate back to the shape of ``orig``."""
    img = img[np.arange(0, img.shape[0], dr)]
    return cv2.resize(img, (orig.shape[1], orig.shape[0]))


def gk_simple(size, std):
    """2D kernel blurring along rows only: a 1D Gaussian in the centre column."""
    kernel = np.zeros((size, size))
    kernel[:, size // 2] = signal.windows.gaussian(size, std)
    return kernel


def gaussian_kernels(kernel_num, size=51, first_std=3, std_step=2):
    """Axial blur kernels of increasing width (std = 3, 5, 7, ...)."""
    stds = first_std + std_step * np.arange(kernel_num)
    return [gk_simple(size, std) for std in stds]


def _degrade(raw_slice, kernel, downsample_rate):
    conved = signal.fftconvolve(raw_slice, kernel, mode='same')
    conved = resize(conved, downsample_rate, raw_slice)
    assert conved.shape == raw_slice.shape
    return conved


def _write_pairs(gt_dir, lq_dir, gt_slices, lq_stacks, suffix=''):
    """Write one gt/lq pair per slice per kernel as ``{kernel}_{slice}{suffix}.tiff``."""
    for k_idx, stack in enumerate(lq_stacks):
        stack = normalize(stack)
        for s_idx, lq in enumerate(stack):
            name = f'{k_idx}_{s_idx}{suffix}.tiff'
            tifffile.imwrite(os.path.join(gt_dir, name), gt_slices[s_idx])
            tifffile.imwrite(os.path.join(lq_dir, name), lq)


def semi_synthetic_creation(raw_tif_pth, save_pth, kernel_num=3, project_depth=5,
                            downsample_rate=5):
    """Create semi-synthetic training pairs from the lateral (XY) slices of a stack.

    Each lateral slice (and each max-intensity projection over ``project_depth``
    slices) is blurred along one axis by ``kernel_num`` Gaussian kernels and
    subsampled by ``downsample_rate``, mimicking the axial degradation.
    """
    raw_data = normalize(tifffile.imread(raw_tif_pth))
    kernels = gaussian_kernels(kernel_num)
    gt_pth = os.path.join(save_pth, 'gt')
    lq_pth = os.path.join(save_pth, 'lq')
    os.makedirs(gt_pth, exist_ok=True)
    os.makedirs(lq_pth, exist_ok=True)

    # single lateral slices
    lq = [[_degrade(s, k, downsample_rate) for s in raw_data] for k in kernels]
    _write_pairs(gt_pth, lq_pth, raw_data, [np.stack(s) for s in lq])

    # max-intensity projections over blocks of `project_depth` slices
    n_proj = raw_data.shape[0] // project_depth - 1
    mips = [np.amax(raw_data[i * project_depth:(i + 1) * project_depth], 0)
            for i in range(n_proj)]
    lq = [[_degrade(m, k, downsample_rate) for m in mips] for k in kernels]
    _write_pairs(gt_pth, lq_pth, mips, [np.stack(s) for s in lq],
                 suffix=f'_{project_depth}')


def denoised_semi_synthetic_creation(input_pth, output_pth, kernel_num=3,
                                     downsample_rate=5):
    """Same as `semi_synthetic_creation` but from a folder of denoised lateral slices.

    Writes ``denoised_gt_train/`` and ``denoised_lq_train/`` under ``output_pth``.
    """
    kernels = gaussian_kernels(kernel_num)
    gt_pth = os.path.join(output_pth, 'denoised_gt_train')
    lq_pth = os.path.join(output_pth, 'denoised_lq_train')
    os.makedirs(gt_pth, exist_ok=True)
    os.makedirs(lq_pth, exist_ok=True)

    raw = [read_slice(os.path.join(input_pth, f'{i}.tiff'))
           for i in range(len(os.listdir(input_pth)))]
    gt = normalize(np.stack(raw))
    lq = [np.stack([_degrade(s, k, downsample_rate) for s in raw]) for k in kernels]
    _write_pairs(gt_pth, lq_pth, gt, lq)


def generate_raw_data(raw_pth, save_pth, dr, crop=None):
    """Slice a raw stack into upsampled axial views for restoration.

    Args:
        raw_pth (str): Path to the (Z, Y, X) stack.
        save_pth (str): Output folder; ``test_xz``, ``test_yz`` and ``test_xy`` are created.
        dr (int): Axial upsampling factor.
        crop (tuple[slice] | None): Optional sub-volume, e.g.
            ``np.s_[:80, :400, :500]`` (the region used in the original demo).
    """
    raw_data = normalize(tifffile.imread(raw_pth))
    assert raw_data.ndim == 3, f'expected a 3D stack, got shape {raw_data.shape}'
    if crop is not None:
        raw_data = raw_data[crop]
    z, y, x = raw_data.shape
    path_xz = os.path.join(save_pth, 'test_xz')
    path_yz = os.path.join(save_pth, 'test_yz')
    path_xy = os.path.join(save_pth, 'test_xy')
    for p in (path_xz, path_yz, path_xy):
        os.makedirs(p, exist_ok=True)

    for idx in range(x):
        view = cv2.resize(raw_data[:, :, idx], (y, z * dr))
        tifffile.imwrite(os.path.join(path_xz, f'{idx}.tiff'), view)
    for idx in range(y):
        view = cv2.resize(raw_data[:, idx, :], (x, z * dr))
        tifffile.imwrite(os.path.join(path_yz, f'{idx}.tiff'), view)
    for idx in range(z):
        tifffile.imwrite(os.path.join(path_xy, f'{idx}.tiff'), raw_data[idx])
    return raw_data.shape


def generate_zs_dataset(input_pth, num_samples=10, half_size=100, seed=None):
    """Sample centre crops of training pairs for the surgeon's zero-shot metrics.

    Reads ``gt/`` and ``lq/`` under ``input_pth`` and writes 8-bit crops of
    size ``2 * half_size`` to ``zs_gt/`` and ``zs_lq/`` in the same folder.
    """
    gt_pth = os.path.join(input_pth, 'gt')
    lq_pth = os.path.join(input_pth, 'lq')
    out = {k: os.path.join(input_pth, f'zs_{k}') for k in ('gt', 'lq')}
    for p in out.values():
        os.makedirs(p, exist_ok=True)

    file_names = sorted(os.listdir(gt_pth))
    rng = random.Random(seed)
    d = half_size
    for file in rng.sample(file_names, min(num_samples, len(file_names))):
        for key, src in (('gt', gt_pth), ('lq', lq_pth)):
            img = tifffile.imread(os.path.join(src, file))
            cy, cx = img.shape[0] // 2, img.shape[1] // 2
            img = img[max(cy - d, 0):cy + d, max(cx - d, 0):cx + d]
            img = (img / img.max() * 255).astype(np.uint8)
            tifffile.imwrite(os.path.join(out[key], file), img)


def read_slice(path):
    """Read a restored slice as float64; 3-channel outputs are averaged."""
    img = tifffile.imread(path).astype(np.float64)
    return img.mean(-1) if img.ndim == 3 else img


def _read_folder(folder):
    n = len(os.listdir(folder))
    return np.stack([read_slice(os.path.join(folder, f'{i}.tiff'))
                     for i in tqdm(range(n), desc=os.path.basename(folder))])


def fuse_views(rec_pth_xz, rec_pth_yz):
    """Average the restored XZ and YZ views into one (Z*dr, Y, X) volume."""
    xz = _read_folder(rec_pth_xz)                  # (X, Z*dr, Y)
    yz = np.swapaxes(_read_folder(rec_pth_yz), 0, 2)  # (X, Z*dr, Y)
    return ((xz + yz) / 2).transpose(1, 2, 0)


def restore_volume(raw_pth, rec_pth_xz, rec_pth_yz):
    """Original API: returns (xz, yz, average), each indexed (X, Z*dr, Y).

    ``raw_pth`` is unused and kept for backwards compatibility; prefer `fuse_views`.
    """
    xz = _read_folder(rec_pth_xz)
    yz = _read_folder(rec_pth_yz)
    return xz, yz, (np.swapaxes(yz, 0, 2) + xz) / 2


def adjust_contrast(image, p_low=2, p_high=98.8):
    """Percentile contrast stretch for display."""
    v_min, v_max = np.percentile(image, (p_low, p_high))
    return exposure.rescale_intensity(image, in_range=(v_min, v_max))
