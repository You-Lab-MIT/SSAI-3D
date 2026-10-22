"""Sparse fine-tuning of the pre-trained network on semi-synthetic pairs."""
import datetime
import logging
import math
import os
import time
from os import path as osp

import torch

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
from ssai3d.config import load_opt


def freeze_layers(net, rank, top_k=20, min_numel=50000, n_output=20, n_edge=30):
    """Leave only the surgeon-selected parts of ``net`` trainable.

    Unfreezes, in parameter order:
      * parameters following each of the ``top_k`` best-ranked large tensors
        (large = more than ``min_numel`` elements, counted from 1);
      * the ``n_output`` parameters before the last one;
      * the first and last ``n_edge`` large tensors.
    """
    params = [p for _, p in net.named_parameters()]
    for p in params:
        p.requires_grad = False

    selected = [layer for layer, _ in rank[:top_k]]
    count, large = 0, []
    for idx, p in enumerate(params):
        if p.numel() > min_numel:
            count += 1
            large.append(idx)
        if count in selected:
            p.requires_grad = True

    last = len(params) - 1
    keep = set(range(last - n_output, last)) | set(large[:n_edge]) | set(large[-n_edge:])
    for idx in keep:
        params[idx].requires_grad = True
    return net


def _latest_state(states_dir):
    if not osp.isdir(states_dir):
        return None
    states = [f for f in os.listdir(states_dir) if f.endswith('.state')]
    if not states:
        return None
    return osp.join(states_dir, f'{max(int(f[:-6]) for f in states)}.state')


def _init_loggers(opt):
    """Console logger plus a log file for this run.

    The file handler is added explicitly: get_root_logger only attaches one when
    the logger is first created, which is usually earlier (e.g. by the surgeon).
    """
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = None
    if opt['logger'].get('use_tb_logger'):
        tb_logger = init_tb_logger(log_dir=osp.join(opt['path']['experiments_root'], 'tb'))
    return logger, tb_logger, file_handler


def _train_loader(opt, logger):
    dataset_opt = opt['datasets']['train']
    train_set = create_dataset(dataset_opt)
    enlarge = dataset_opt.get('dataset_enlarge_ratio', 1)
    sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], enlarge)
    loader = create_dataloader(train_set, dataset_opt, num_gpu=opt['num_gpu'],
                               dist=False, sampler=sampler, seed=opt['manual_seed'])
    iters_per_epoch = math.ceil(len(train_set) * enlarge / dataset_opt['batch_size_per_gpu'])
    total_iters = int(opt['train']['total_iter'])
    logger.info(f'Training images: {len(train_set)}, iters per epoch: {iters_per_epoch}, '
                f'total iters: {total_iters}.')
    return loader, sampler, total_iters


def finetune(rank, gt_dir, lq_dir, pretrained, experiments_root, iters=1000, lr=1e-4,
             gt_size=500, batch_size=1, num_workers=8, save_every=500, resume=False,
             freeze_kwargs=None, device=None):
    """Fine-tune ``pretrained`` on paired slices, training only the ranked layers.

    Args:
        rank (list): Output of `ssai3d.surgeon.rank_layers`.
        gt_dir, lq_dir (str): Folders of matching ground-truth / degraded slices.
        pretrained (str): Pre-trained network weights.
        experiments_root (str): Output folder; the final weights are written to
            ``<experiments_root>/models/net_g_latest.pth``.
        resume (bool): Continue from the latest state in ``training_states/``.

    Returns:
        str: Path of the final weights.
    """
    overrides = {
        'datasets': {'train': {'dataroot_gt': gt_dir, 'dataroot_lq': lq_dir,
                               'gt_size': gt_size, 'batch_size_per_gpu': batch_size,
                               'num_worker_per_gpu': num_workers}},
        'path': {'pretrain_network_g': str(pretrained)},
        'train': {'total_iter': iters, 'optim_g': {'lr': lr},
                  'scheduler': {'T_max': iters}},
        'logger': {'save_checkpoint_freq': save_every},
    }
    opt = load_opt('finetune', is_train=True, experiments_root=experiments_root,
                   overrides=overrides, device=device)
    torch.backends.cudnn.benchmark = True

    resume_state = None
    if resume:
        state_file = _latest_state(opt['path']['training_states'])
        if state_file:
            opt['path']['resume_state'] = state_file
            resume_state = torch.load(state_file, map_location='cpu')
    if resume_state is None:
        make_exp_dirs(opt)

    logger, tb_logger, file_handler = _init_loggers(opt)
    train_loader, train_sampler, total_iters = _train_loader(opt, logger)

    model = create_model(opt)
    start_epoch, current_iter = 0, 0
    if resume_state:
        check_resume(opt, resume_state['iter'])
        model.resume_training(resume_state)
        start_epoch, current_iter = resume_state['epoch'], resume_state['iter']
        logger.info(f'Resuming training from epoch {start_epoch}, iter {current_iter}.')
    model.net_g = freeze_layers(model.net_g, rank, **(freeze_kwargs or {}))

    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    prefetcher = CPUPrefetcher(train_loader)

    logger.info(f'Start training from epoch {start_epoch}, iter {current_iter}.')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        while train_data is not None:
            data_time = time.time() - data_time
            current_iter += 1
            if current_iter > total_iters:
                break
            model.update_learning_rate(current_iter,
                                       warmup_iter=opt['train'].get('warmup_iter', -1))
            model.feed_data(train_data, is_val=False)
            model.optimize_parameters(current_iter, tb_logger)
            iter_time = time.time() - iter_time
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters,
                            'lrs': model.get_current_learning_rate(),
                            'time': iter_time, 'data_time': data_time}
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)
            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        epoch += 1

    elapsed = datetime.timedelta(seconds=int(time.time() - start_time))
    logger.info(f'End of training. Time consumed: {elapsed}')
    model.save(epoch=-1, current_iter=-1)
    if tb_logger:
        tb_logger.close()
    logger.removeHandler(file_handler)
    file_handler.close()
    return osp.join(opt['path']['models'], 'net_g_latest.pth')
