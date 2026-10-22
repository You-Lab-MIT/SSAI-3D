"""Build BasicSR option dicts from the bundled YAML configs, without argparse."""
import random
from os import path as osp
from pathlib import Path

import torch

from basicsr.utils import set_random_seed
from basicsr.utils.options import parse

CONFIG_DIR = Path(__file__).parent / 'configs'


def config_path(name):
    """Resolve a bundled config name ('finetune', 'denoise') or return a path as-is."""
    bundled = CONFIG_DIR / f'{name}.yml'
    return str(bundled) if bundled.exists() else str(name)


def _merge(base, overrides):
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_opt(config='finetune', is_train=False, experiments_root=None, overrides=None,
             device=None):
    """Load an option dict for a single, non-distributed process.

    Args:
        config (str): Bundled config name or path to a YAML file.
        is_train (bool): Training (creates optimizer/loss settings) or inference.
        experiments_root (str | None): Folder for checkpoints, states and logs.
            Defaults to ``./experiments/<name>`` in the working directory.
        overrides (dict | None): Nested values merged into the parsed options.
        device (torch.device | None): Device from `ssai3d.gpu.setup_device`;
            None uses the current CUDA device if there is one.
    """
    opt = parse(config_path(config), is_train=is_train)
    if overrides:
        _merge(opt, overrides)

    root = osp.abspath(experiments_root or osp.join('experiments', opt['name']))
    opt['path']['experiments_root'] = root
    if is_train:
        opt['path'].update(models=osp.join(root, 'models'),
                           training_states=osp.join(root, 'training_states'),
                           log=root,
                           visualization=osp.join(root, 'visualization'))
    else:
        opt['path'].update(results_root=root, log=root,
                           visualization=osp.join(root, 'visualization'))

    opt['dist'] = False
    opt['rank'], opt['world_size'] = 0, 1
    # One device at most: the one selected with ssai3d.gpu.setup_device.
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt['num_gpu'] = 1 if torch.device(device).type == 'cuda' else 0
    if opt.get('manual_seed') is None:
        opt['manual_seed'] = random.randint(1, 10000)
    set_random_seed(opt['manual_seed'])
    return opt
