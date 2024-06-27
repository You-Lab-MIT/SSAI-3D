import torch
import os 
from basicsr.models import create_model
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite
import argparse
from basicsr.utils.options import parse
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils import set_random_seed
import json

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str,default =\
        '/home/youlab/Desktop/workspace/jiashu/SSAI3D/options/train/custom/ZEST.yml', required=False, help='Path to option YAML file.')
        
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--input_path', type=str, required=False, help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, required=False, help='The path to the output image. For single image inference only.')
    args, unknown = parser.parse_known_args()
    opt = parse(args.opt, is_train=is_train)
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)
    print(args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }

    return opt

def folder_deconvolve(input_pth, output_pth, model_pth):
    os.makedirs(output_pth, exist_ok = True)
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()
    file_client = FileClient('disk')
    opt['path']['pretrain_network_g'] = model_pth
    model = create_model(opt)
    for n, p in model.net_g.named_parameters():
        p.requires_grad = False
    for count, img_path in enumerate(os.listdir(input_pth)):
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
        imwrite(sr_img, os.path.join(output_pth, img_path))
        if count % 20 == 0:
            print(f'{count}th inference {img_path} .. finished. saved to {output_pth}')
    return 

def options(sample):
    configs = None
    with open('../basicsr/demo_/configs.json') as f:
        configs = json.load(f)
    model_pth = configs[sample]["model_pth"]
    data_pth = configs[sample]["dataset_pth"]
    
    return model_pth, data_pth