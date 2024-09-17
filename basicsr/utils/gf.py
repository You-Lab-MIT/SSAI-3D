import numpy as np
import os
import scipy.signal as signal



def gk_simple(filter, std):
    # std = 4
    # filter = 31
    gau_filter_ = signal.gaussian(filter,std)
    gau_filter = np.zeros((filter,filter,filter))
    middle = int(gau_filter_.shape[0]/2)
    gau_filter[:,middle,middle] = gau_filter_
    gau_filter = np.amax(gau_filter[:,::,:],1)
    return gau_filter

def norm_k(kernel):
    kernel = kernel/kernel.sum()
    return kernel

def gk_3d(size, sigma_x, sigma_y, sigma_z):
    """Generate a 3D Gaussian kernel."""
    # Create a grid of (x, y, z) coordinates
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    z = np.linspace(-size // 2, size // 2, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    kernel = (1 / ((2 * np.pi)**1.5 * sigma_x * sigma_y * sigma_z)) * \
             np.exp(-((X**2 / (2 * sigma_x**2)) + (Y**2 / (2 * sigma_y**2)) + (Z**2 / (2 * sigma_z**2))))
    kernel = kernel/kernel.max()
    kernel = kernel * 65535
    kernel = kernel.astype(np.uint16)
    kernel = kernel / kernel.sum()
    return kernel

def create_semi_set(tiff_file):
    x, y, z = tiff_file.shape
    kernels = [
        np.amax(generate_3d_gaussian_kernel(51, 7., .5, .5)[:,::,:]),
        np.amax(generate_3d_gaussian_kernel(51, 6., .5, .5)[:,::,:]),
        np.amax(generate_3d_gaussian_kernel(51, 5., .5, .5)[:,::,:]),
        np.amax(generate_3d_gaussian_kernel(51, 4., .5, .5)[:,::,:])
    ]
    for x_slice_idx in range(x//5):
        x_slice_idx = x_slice_idx * 5
        x_slice = tiff_file[x_slice_idx,:,:]
        for k_index in (kernels):
            semi_data = convolve(x_slice, k_index, mode = 'same')
    return 


if __name__ == '__main__':
    kernel = generate_3d_gaussian_kernel(51, 7., 0.5, 0.5)