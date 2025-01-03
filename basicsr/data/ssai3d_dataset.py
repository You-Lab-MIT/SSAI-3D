import os
import cv2
import numpy as np
import tifffile
from basicsr.utils.gf import *
from basicsr.utils.dataset import normalize, resize
import random
from skimage import exposure
from tqdm import tqdm 
import tifffile

def restore_volume(raw_pth, rec_pth_xz, rec_pth_yz ):
    raw_stack = []
    rec_stack1 = []
    rec_stack2 = []
    dlen1 = len(os.listdir(raw_pth))
    dlen2 = len(os.listdir(rec_pth_yz))

    for i in tqdm(range(dlen1)):
        rec_slice1 = tifffile.imread(os.path.join(rec_pth_xz, f'{i}.tiff'))
        # raw_stack.append(raw_slice)
        rec_stack1.append(rec_slice1)

    for i in tqdm(range(dlen2)):
        rec_slice2 = tifffile.imread(os.path.join(rec_pth_yz, f'{i}.tiff'))
        rec_stack2.append(rec_slice2)

    rec_stack_1 = np.stack(rec_stack1).mean(-1)
    rec_stack_2 = np.stack(rec_stack2).mean(-1)
    rec_stack_1, rec_stack_2 = rec_stack_1.astype(np.float64), rec_stack_2.astype(np.float64)
    tmp = np.swapaxes(rec_stack_2,0,2 )
    res = (tmp + rec_stack_1)/2
    return rec_stack_1, rec_stack_2, res

def denoised_semi_synthetic_creation(input_pth, output_pth,
    kernel_num = 3, downsample_rate = 5,  rotation = False):
    kernel_lst = []
    res_lst = [[] for _ in range(kernel_num)]
    gt_lst = [[] for _ in range(kernel_num)]
    
    for idx, std in enumerate(np.arange(3,101,2)):
        if idx >= kernel_num:
            break
        kernel_lst.append(gk_simple(51, std))
    
    gt_pth = os.path.join(output_pth, 'denoised_gt_train')
    lq_pth = os.path.join(output_pth, 'denoised_lq_train')
    os.makedirs(gt_pth, exist_ok = True)
    os.makedirs(lq_pth, exist_ok = True)

    xy_slices = len(os.listdir(input_pth))
    for slice_idx in range(xy_slices):
        raw_slice = tifffile.imread(os.path.join(input_pth, f'{slice_idx}.tiff')).mean(-1)
        for idx, k in enumerate(kernel_lst):

            conved_slice = signal.fftconvolve(raw_slice, k, mode = 'same')
            conved_slice = resize(conved_slice, downsample_rate, raw_slice)
            res_lst[idx].append(conved_slice)
            gt_lst[idx].append(raw_slice)
            assert conved_slice.shape == raw_slice.shape
            # cv2.imwrite(os.path.join(output_pth,f'{slice_idx}' ), slice)
    lqstacks = [np.stack(s) for s in res_lst]
    gtstacks = [np.stack(s) for s in gt_lst]
    for idx, stack in enumerate(lqstacks):
        lqstack = normalize(stack)
        gtstack = normalize(gtstacks[idx])
        for slice_idx, lq_slice in enumerate(lqstack):
            gt_slice = gtstack[slice_idx]
            tifffile.imwrite(os.path.join(gt_pth, f'{idx}_{slice_idx}.tiff'), gt_slice)
            tifffile.imwrite(os.path.join(lq_pth, f'{idx}_{slice_idx}.tiff'), lq_slice)
    if rotation:
        pass
    return 
def semi_synthetic_creation(raw_tif_pth, save_pth, kernel_num = 7, project_depth = 5, \
    downsample_rate = 5,  rotation = False):

    raw_data = tifffile.imread(raw_tif_pth)
    raw_data = normalize(raw_data)
    kernel_lst = []
    res_lst = [[] for _ in range(kernel_num)]
    z_slices = raw_data.shape[0]
    
    proj_idx = project_depth
    slices = (raw_data.shape[0]//proj_idx) - 1 
    
    for idx, std in enumerate(np.arange(3,101,2)):
        if idx >= kernel_num:
            break
        kernel_lst.append(gk_simple(51, std))

    gt_pth = os.path.join(save_pth, 'gt')
    lq_pth = os.path.join(save_pth, 'lq')
    os.makedirs(gt_pth, exist_ok = True)
    os.makedirs(lq_pth, exist_ok = True)
    for slice_idx in range(z_slices):
        raw_slice = raw_data[slice_idx]
        for idx, k in enumerate(kernel_lst):
            conved_slice = signal.fftconvolve(raw_slice, k, mode = 'same')
            conved_slice = resize(conved_slice, downsample_rate, raw_slice)
            res_lst[idx].append(conved_slice)
            assert conved_slice.shape == raw_slice.shape
    stacks = [np.stack(s) for s in res_lst]

    for idx, stack in enumerate(stacks):
        stack = normalize(stack)
        for slice_idx, slice in enumerate(stack):
            tifffile.imwrite(os.path.join(gt_pth, f'{idx}_{slice_idx}.tiff'), raw_data[slice_idx])
            tifffile.imwrite(os.path.join(lq_pth, f'{idx}_{slice_idx}.tiff'), slice)
    
    res_lst = [[] for _ in range(kernel_num)]
    for slice_idx in range(slices):
        raw_slice = np.amax(raw_data[slice_idx * proj_idx:(slice_idx+1) * proj_idx], 0)
        for idx, k in enumerate(kernel_lst):
            conved_slice = signal.fftconvolve(raw_slice, k, mode = 'same')
            conved_slice = resize(conved_slice, 5, raw_slice)
            res_lst[idx].append(conved_slice)
            assert conved_slice.shape == raw_slice.shape

    stacks = [np.stack(s) for s in res_lst]
    for idx, stack in enumerate(stacks):
        stack = normalize(stack)
        for slice_idx, slice in enumerate(stack):
            tifffile.imwrite(os.path.join(gt_pth, f'{idx}_{slice_idx}_{proj_idx}.tiff'), np.amax(raw_data[slice_idx * proj_idx:(slice_idx+1) * proj_idx], 0))
            tifffile.imwrite(os.path.join(lq_pth, f'{idx}_{slice_idx}_{proj_idx}.tiff'), slice)
    if rotation:
        pass
    return 

# def generate_test(raw_pth):
#     raw_data = tifffile.imread(raw_pth)
#     raw_data = normalize(raw_data)
#     path_xz = os.path.join(save_pth,'test_xz')
#     path_yz = os.path.join(save_pth,'test_yz')

#     os.makedirs(path_xz, exist_ok=True)
#     os.makedirs(path_yz, exist_ok=True)
    
#     assert len(raw_data.shape) == 3
#     xz_len = raw_data.shape[1]
#     yz_len = raw_data.shape[-1]

#     for idx in range(yz_len):
#         slice = raw_data[:,:,idx]
#         slice = cv2.resize(slice, (raw_data.shape[1], raw_data.shape[0]*dr))
#         cv2.imwrite(os.path.join(path_xz, f'{idx}.tiff'), slice)
    
#     for idx in range(xz_len):
#         slice = raw_data[:,idx,:]
#         slice = cv2.resize(slice, (raw_data.shape[-1], raw_data.shape[0]*dr))
#         cv2.imwrite(os.path.join(path_yz, f'{idx}.tiff'), slice)    
#     pass


def generate_raw_data(raw_pth, save_pth, dr):
    raw_data = tifffile.imread(raw_pth)
    raw_data = normalize(raw_data)
    print(raw_data.dtype)
    path_xz = os.path.join(save_pth,'test_xz')
    path_yz = os.path.join(save_pth,'test_yz')
    path_xy = os.path.join(save_pth,'test_xy')

    raw_data = raw_data[:80,:400,:500]
    os.makedirs(path_xz, exist_ok=True)
    os.makedirs(path_yz, exist_ok=True)
    os.makedirs(path_xy, exist_ok=True)
    
    assert len(raw_data.shape) == 3
    xz_len = raw_data.shape[1]
    yz_len = raw_data.shape[-1]
    xy_len = raw_data.shape[0]

    for idx in range(yz_len):
        slice = raw_data[:,:,idx]
        slice = cv2.resize(slice, (raw_data.shape[1], raw_data.shape[0]*dr))
        tifffile.imwrite(os.path.join(path_xz, f'{idx}.tiff'), slice)
    

    for idx in range(xy_len):
        slice = raw_data[:,:,idx]
        slice = cv2.resize(slice, (raw_data.shape[1], raw_data.shape[0]*dr))
        tifffile.imwrite(os.path.join(path_xy, f'{idx}.tiff'), slice)

    for idx in range(xz_len):
        slice = raw_data[:,idx,:]
        slice = cv2.resize(slice, (raw_data.shape[-1], raw_data.shape[0]*dr))
        tifffile.imwrite(os.path.join(path_yz, f'{idx}.tiff'), slice)
    pass

def generate_zs_dataset(input_pth):
    os.makedirs(os.path.join(input_pth, 'zs_lq'), exist_ok = True)
    os.makedirs(os.path.join(input_pth, 'zs_gt'), exist_ok = True)
    gt_pth = os.path.join(input_pth, 'gt')
    lq_pth = os.path.join(input_pth, 'lq')
    file_names = os.listdir(gt_pth)
    len_pth = len(file_names)
    gt_lst = []
    lq_lst = []

    d = 100
    files = random.sample(file_names, 10)
    for file in files:
        gt_s = tifffile.imread(os.path.join(gt_pth, file))
        lq_s = tifffile.imread(os.path.join(lq_pth, file))
        x = lq_s.shape[0]
        y = lq_s.shape[1]
        gt_s = gt_s[x-d:x+d, y-d:y+d]
        lq_s = lq_s[x-d:x+d, y-d:y+d]
        gt_s = gt_s/gt_s.max()
        gt_s = gt_s * 255
        gt_s = gt_s.astype(np.uint8)

        lq_s = lq_s/lq_s.max()
        lq_s = lq_s * 255
        lq_s = lq_s.astype(np.uint8)
        tifffile.imwrite(f'./demo_dataset/zs_lq/{file}', lq_s)
        tifffile.imwrite(f'./demo_dataset/zs_gt/{file}', gt_s)


def adjust_contrast(image, p_low=2, p_high=98.8):
    v_min, v_max = np.percentile(image, (p_low, p_high))  # Calculate intensity bounds
    return exposure.rescale_intensity(image, in_range=(v_min, v_max))
