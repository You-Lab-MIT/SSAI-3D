"""End-to-end SSAI-3D run driven by one YAML config (see configs/example_run.yml)."""
import copy
import json
import logging
import os
import shutil
import time
from os import path as osp

import numpy as np
import tifffile
import torch
import yaml

from ssai3d import data
from ssai3d.gpu import peak_memory_gb, setup_device
from ssai3d.restore import restore
from ssai3d.surgeon import rank_layers
from ssai3d.train import finetune

logger = logging.getLogger('ssai3d')

DEFAULTS = {
    'name': 'sample',
    'raw': None,
    'out_dir': None,
    'mode': 'finetune',          # finetune | pretrained
    'downsample': 5,             # axial / lateral sampling ratio
    'gpu': 0,                    # CUDA index, or -1 for CPU
    'max_mem_gb': None,          # total GPU memory budget for this process
    'crop': None,                # [[z0, z1], [y0, y1], [x0, x1]] restored sub-volume
    'weights': {
        'pretrained': None,      # finetune: starting network
        'model': None,           # pretrained: network applied directly
        'denoise': None,         # optional denoising network
    },
    'prepare': {'kernel_num': 3, 'project_depth': 5, 'zs_samples': 10,
                'zs_half_size': 100, 'seed': 0},
    'surgeon': {'num_samples': 2},
    'train': {'iters': 1000, 'lr': 1.0e-4, 'gt_size': 500, 'batch_size': 1,
              'num_workers': 8, 'save_every': 500},
    'restore': {'tile': None},
    'output': {'keep_intermediate': True},
}

STEPS = {
    'finetune': ['prepare', 'rank', 'train', 'restore', 'fuse'],
    'pretrained': ['slice', 'restore', 'fuse'],
}


def _merge(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    """Read a run config; relative paths are resolved against its folder."""
    with open(path) as f:
        cfg = _merge(copy.deepcopy(DEFAULTS), yaml.safe_load(f) or {})
    base = osp.dirname(osp.abspath(path))

    def resolve(p):
        return None if p is None else osp.normpath(osp.join(base, osp.expanduser(p)))

    cfg['raw'] = resolve(cfg['raw'])
    cfg['out_dir'] = resolve(cfg['out_dir'] or osp.join('runs', cfg['name']))
    cfg['weights'] = {k: resolve(v) for k, v in cfg['weights'].items()}
    return cfg


class Run:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out = cfg['out_dir']
        self.data = osp.join(self.out, 'data')
        self.done = osp.join(self.out, '.done')
        os.makedirs(self.done, exist_ok=True)
        self.metrics_file = osp.join(self.out, 'metrics.json')
        self.metrics = {}
        if osp.exists(self.metrics_file):
            with open(self.metrics_file) as f:
                self.metrics = json.load(f)
        self.device = setup_device(cfg['gpu'], cfg['max_mem_gb'])

    # -- helpers -------------------------------------------------------------
    @property
    def denoise(self):
        return self.cfg['weights']['denoise']

    @property
    def crop(self):
        c = self.cfg['crop']
        return None if c is None else tuple(slice(*a) for a in c)

    def _rank_file(self):
        return osp.join(self.out, 'rank.json')

    def _model_file(self):
        if self.cfg['mode'] == 'pretrained':
            return self.cfg['weights']['model']
        return osp.join(self.out, 'train', 'models', 'net_g_latest.pth')

    # -- steps ---------------------------------------------------------------
    def slice(self):
        shape = data.generate_raw_data(self.cfg['raw'], self.data, self.cfg['downsample'],
                                       crop=self.crop)
        logger.info(f'Restoring sub-volume of shape {shape} (Z, Y, X).')

    def prepare(self):
        p = self.cfg['prepare']
        data.semi_synthetic_creation(self.cfg['raw'], self.data, kernel_num=p['kernel_num'],
                                     project_depth=p['project_depth'],
                                     downsample_rate=self.cfg['downsample'])
        self.slice()
        data.generate_zs_dataset(self.data, num_samples=p['zs_samples'],
                                 half_size=p['zs_half_size'], seed=p['seed'])
        if self.denoise:
            restore(osp.join(self.data, 'test_xy'), osp.join(self.data, 'denoised_gt'),
                    self.denoise, denoise=True, tile=self.cfg['restore']['tile'],
                    device=self.device)
            data.denoised_semi_synthetic_creation(
                osp.join(self.data, 'denoised_gt'), self.data,
                kernel_num=p['kernel_num'], downsample_rate=self.cfg['downsample'])

    def rank(self):
        rank = rank_layers(self.cfg['weights']['pretrained'], self.data,
                           num_samples=self.cfg['surgeon']['num_samples'])
        with open(self._rank_file(), 'w') as f:
            json.dump(rank, f)

    def train(self):
        with open(self._rank_file()) as f:
            rank = json.load(f)
        pairs = ('denoised_gt_train', 'denoised_lq_train') if self.denoise else ('gt', 'lq')
        t = self.cfg['train']
        finetune(rank, osp.join(self.data, pairs[0]), osp.join(self.data, pairs[1]),
                 self.cfg['weights']['pretrained'], osp.join(self.out, 'train'),
                 iters=t['iters'], lr=t['lr'], gt_size=t['gt_size'],
                 batch_size=t['batch_size'], num_workers=t['num_workers'],
                 save_every=t['save_every'], device=self.device)

    def restore(self):
        tile = self.cfg['restore']['tile']
        for view in ('xz', 'yz'):
            src = osp.join(self.data, f'test_{view}')
            if self.denoise:
                denoised = osp.join(self.data, f'denoised_{view}')
                restore(src, denoised, self.denoise, denoise=True, tile=tile,
                        device=self.device)
                src = denoised
            restore(src, osp.join(self.out, 'restored', view), self._model_file(), tile=tile,
                    device=self.device)

    def fuse(self):
        vol = data.fuse_views(osp.join(self.out, 'restored', 'xz'),
                              osp.join(self.out, 'restored', 'yz'))
        path = osp.join(self.out, f"{self.cfg['name']}_restored.tif")
        tifffile.imwrite(path, vol.astype(np.float32), imagej=True,
                         metadata={'axes': 'ZYX'})
        logger.info(f'Wrote {path} with shape {vol.shape} (Z, Y, X).')
        if not self.cfg['output']['keep_intermediate']:
            for d in (self.data, osp.join(self.out, 'restored')):
                shutil.rmtree(d, ignore_errors=True)

    # -- driver --------------------------------------------------------------
    def run(self, steps=None, force=False):
        steps = steps or STEPS[self.cfg['mode']]
        for step in steps:
            marker = osp.join(self.done, step)
            if osp.exists(marker) and not force:
                logger.info(f'[{step}] already done, skipping (use --force to redo).')
                continue
            logger.info(f'[{step}] starting')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            getattr(self, step)()
            self.metrics[step] = {'seconds': round(time.time() - t0, 1),
                                  'peak_gpu_gb': round(peak_memory_gb(), 2)}
            logger.info(f'[{step}] done: {self.metrics[step]}')
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            open(marker, 'w').close()


def run(config_path, steps=None, force=False):
    cfg = load_config(config_path)
    if cfg['raw'] is None:
        raise ValueError('`raw` (path to the 3D TIFF) is required.')
    os.makedirs(cfg['out_dir'], exist_ok=True)
    with open(osp.join(cfg['out_dir'], 'config.yml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    Run(cfg).run(steps, force)
    return cfg['out_dir']
