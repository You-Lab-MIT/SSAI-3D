import torch.nn as nn
import torch
import random
import cv2
import numpy as np
import os
from basicsr.utils import parse_options, FileClient, img2tensor
from basicsr.utils import get_grad_norm_arr,compute_snip_per_weight, compute_grasp_per_weight, \
    compute_fisher_per_weight, compute_plain_per_weight,compute_synflow_per_weight 
from basicsr.models import create_model
# from basicsr.utils.file_client import FileClient


from basicsr.trainer.trainer import SurgeonTrainer
from basicsr.trainer.surgeon import Surgeon

__all__ = ['SurgeonTrainer', 'Surgeon']

if __name__ == '__main__':
    configs = EasyDict({
    'model_path' : f'../experiments/pretrained/endometrium/CH1/net_g_latest.pth',
    'lq_pth': f'../datasets/final/beads/2.0/lq',
    'gt_pth': f'../final/beads/gt',
    'lr': 1e-4
    })
    operation = Surgeon(configs)
    operation.get_zeroshot_information()
    dimensions = 8
    surgeon_trainer = SurgeonTrainer(dimensions)
    elementwise_input = operation.input_dict.copy()
    rank = surgeon_trainer.forward_all(elementwise_input)
    import ipdb; ipdb.set_trace()