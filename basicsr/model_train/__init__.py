import torch.nn as nn
import torch
import random
import cv2
import datetime

import numpy as np
import os
from os import path as osp
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
import time
from basicsr.utils import parse_options, FileClient, img2tensor
from basicsr.utils import get_grad_norm_arr,compute_snip_per_weight, compute_grasp_per_weight, \
    compute_fisher_per_weight, compute_plain_per_weight,compute_synflow_per_weight 
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite
from tqdm import tqdm
# from basicsr.utils.file_client import FileClient

from basicsr.model_train.utils import create_train_val_dataloader, init_loggers, options, parse_options, parse_options_denoise
from basicsr.model_train.net_utils import freeze_layer_v3
__all__ = ['create_train_val_dataloader',
            'init_loggers', 
            'options',
            'parse_options', 'restore', 'parse_options_denoise']




def trainer_train(freeze_layers, train_data_pth,train_data_pth_lq=None ):
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)
    if train_data_pth_lq == None:
        opt['datasets']['train']['dataroot_gt'] = f'{train_data_pth}/gt'
        opt['datasets']['train']['dataroot_lq'] = f'{train_data_pth}/lq'
    else:
        opt['datasets']['train']['dataroot_gt'] = train_data_pth
        opt['datasets']['train']['dataroot_lq'] = train_data_pth_lq
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        print('!!!!!! resume state .. ', states, state_folder_path)
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = f'{state_folder_path}/{max_state_file}'
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):
    epoch = start_epoch    
    if opt['ZEST']:
        logger.info('*************************************************************************************')
        # model = zest(model)
        logger.info('Surgeon Activated')
        logger.info('*************************************************************************************')
    
    model.net_g = freeze_layer_v3(model.net_g, freeze_layers)
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data, is_val=False)
            result_code = model.optimize_parameters(current_iter, tb_logger)
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                # print('msg logger .. ', current_iter)
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0 or current_iter == 1000:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)
            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
            
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)
    if tb_logger:
        tb_logger.close()

def restore(input_pth, output_pth, model_pth, denoise = False):
    os.makedirs(output_pth, exist_ok = True)
    if denoise:
        opt = parse_options_denoise(is_train=False)
    else:
        opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()
    file_client = FileClient('disk')
    opt['path']['pretrain_network_g'] = model_pth
    model = create_model(opt)
    for n, p in model.net_g.named_parameters():
        p.requires_grad = False
    total = len(os.listdir(input_pth))
    with tqdm(total = total) as pbar:
        for count, img_path in (enumerate(os.listdir(input_pth))):
            img_bytes = file_client.get(os.path.join(input_pth, img_path), None)
            img = imfrombytes(img_bytes, float32=True)
            img = img2tensor(img, bgr2rgb=True, float32=True)
            opt['dist'] = False
            model.feed_data(data={'lq': img.unsqueeze(dim=0)})
            if model.opt['val'].get('grids', False):
                model.grids()
            if model.opt['val'].get('grids', False):
                model.grids_inverse()
            model.test()
            visuals = model.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if sr_img.shape == 4:
                sr_img = sr_img.mean(-1)
            imwrite(sr_img, os.path.join(output_pth, img_path))
            pbar.update(1)
    return 


if __name__ == '__main__':
    rank = [(60, 0.06589828431606293),
        (218, 0.06676021218299866),
        (169, 0.06754997372627258),
        (183, 0.06897436827421188),
        (30, 0.06897670775651932),
        (226, 0.07017166167497635),
        (12, 0.07034114003181458),
        (93, 0.07054248452186584),
        (137, 0.0707058534026146),
        (17, 0.0707860216498375),
        (21, 0.07085736840963364),
        (201, 0.07120496034622192),
        (15, 0.07125891745090485),
        (7, 0.07217205315828323),
        (34, 0.07227000594139099)]
    data_pth = '/home/youlab/Desktop/workspace/kunzan/SSAI-3D/demo/demo_dataset'
    trainer_train(rank, data_pth)