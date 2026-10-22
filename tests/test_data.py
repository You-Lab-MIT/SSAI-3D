import os

import numpy as np
import pytest
import tifffile
from scipy import signal

from ssai3d import data


@pytest.fixture
def stack(tmp_path):
    rng = np.random.default_rng(0)
    vol = rng.integers(1, 4000, size=(12, 40, 30)).astype(np.uint16)
    path = tmp_path / 'raw.tif'
    tifffile.imwrite(path, vol)
    return path, vol


def test_gk_simple_matches_original():
    # original: 1D Gaussian placed in a 3D array, max-projected along axis 1
    size, std = 51, 5
    g = np.zeros((size, size, size))
    g[:, size // 2, size // 2] = signal.windows.gaussian(size, std)
    np.testing.assert_array_equal(data.gk_simple(size, std), np.amax(g, 1))


def test_generate_raw_data_views(stack, tmp_path):
    path, vol = stack
    out = tmp_path / 'slices'
    assert data.generate_raw_data(path, out, dr=3) == vol.shape
    z, y, x = vol.shape
    assert len(os.listdir(out / 'test_xz')) == x
    assert len(os.listdir(out / 'test_yz')) == y
    assert len(os.listdir(out / 'test_xy')) == z
    assert tifffile.imread(out / 'test_xz' / '0.tiff').shape == (z * 3, y)
    assert tifffile.imread(out / 'test_yz' / '0.tiff').shape == (z * 3, x)
    np.testing.assert_array_equal(tifffile.imread(out / 'test_xy' / '2.tiff'),
                                  data.normalize(vol)[2])


def test_generate_raw_data_crop(stack, tmp_path):
    path, _ = stack
    shape = data.generate_raw_data(path, tmp_path / 's', dr=2, crop=np.s_[:5, :10, :20])
    assert shape == (5, 10, 20)


def test_semi_synthetic_pairs(stack, tmp_path):
    path, vol = stack
    data.semi_synthetic_creation(path, tmp_path, kernel_num=2, project_depth=3,
                                 downsample_rate=2)
    gt, lq = sorted(os.listdir(tmp_path / 'gt')), sorted(os.listdir(tmp_path / 'lq'))
    n_proj = vol.shape[0] // 3 - 1
    assert gt == lq
    assert len(gt) == 2 * (vol.shape[0] + n_proj)
    assert '1_0_3.tiff' in gt
    np.testing.assert_array_equal(tifffile.imread(tmp_path / 'gt' / '0_4.tiff'),
                                  data.normalize(vol)[4])


def test_zs_dataset_centre_crop(stack, tmp_path):
    path, _ = stack
    data.semi_synthetic_creation(path, tmp_path, kernel_num=1, project_depth=3,
                                 downsample_rate=2)
    data.generate_zs_dataset(tmp_path, num_samples=4, half_size=10, seed=1)
    files = sorted(os.listdir(tmp_path / 'zs_gt'))
    assert len(files) == 4 and files == sorted(os.listdir(tmp_path / 'zs_lq'))
    crop = tifffile.imread(tmp_path / 'zs_gt' / files[0])
    assert crop.shape == (20, 20) and crop.dtype == np.uint8


def test_fuse_views_orientation(tmp_path):
    # restored views of a volume v with shape (Z', Y, X):
    # xz[x] = v[:, :, x], yz[y] = v[:, y, :]; fusing them must give back v
    v = np.random.default_rng(0).random((6, 4, 5)).astype(np.float32)
    for view, n, take in (('xz', 5, lambda i: v[:, :, i]), ('yz', 4, lambda i: v[:, i, :])):
        os.makedirs(tmp_path / view)
        for i in range(n):
            tifffile.imwrite(tmp_path / view / f'{i}.tiff', take(i))
    fused = data.fuse_views(tmp_path / 'xz', tmp_path / 'yz')
    np.testing.assert_allclose(fused, v, rtol=1e-6)
    _, _, av = data.restore_volume(None, tmp_path / 'xz', tmp_path / 'yz')
    np.testing.assert_allclose(av.transpose(1, 2, 0), fused)
