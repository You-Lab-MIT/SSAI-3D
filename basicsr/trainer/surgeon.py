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

class Surgeon(object):
    def __init__(self, configs):
        model_path = configs.model_path
        opt = parse_options(is_train=True)
        opt['num_gpu'] = 0
        # torch.cuda.device_count()
        file_client = FileClient('disk')
        opt['path']['pretrain_network_g'] = model_path
        self.model = create_model(opt)
        self.gt_pth = configs.gt_pth
        self.lq_pth = configs.lq_pth
        self.lr = configs.lr
        
        self.val_criterion = torch.nn.CrossEntropyLoss()
        self.weight_dict = {}
        self.grads_dict = {}
        self.initialize_saver()
        self.zs_dict = {}
        self.all_dict = {}


    def initialize_saver(self):
        for idx in range(227):
            self.weight_dict[idx] = []
            self.grads_dict[idx] = []

    def create_dataset(self, tiff_path):
        pass
    
    def get_zeroshot_information(self):

        self.get_static()
        self.get_zeroshot()
        self.organize()

    def organize(self):
        proxy_dict = {}
        min_max_dict = {}
        # import ipdb; ipdb.set_trace()
        # bad_ids = self.bad_ids()
        for k, v in self.zs_dict.items():
            lst = []
            for el in v:
                lst.append(el.mean().item())
            # for ids in bad_ids:
            #     del lst[ids]
            min_max_dict[k] = (min(lst), max(lst))

        for k, v in self.zs_dict.items():
            proxy_dict[k] = {}
            for idx, val in enumerate(v):
                try:
                    proxy_dict[k][idx+1] = (val.item() - min_max_dict[k][0]) / (min_max_dict[k][1] - min_max_dict[k][0])
                except:
                    proxy_dict[k][idx+1] = (val.mean().item() - min_max_dict[k][0]) / (min_max_dict[k][1] - min_max_dict[k][0])
        
        mm_d = {
            'mean' : {},
            'var' : {}}
        
        mean_mm = []
        var_mm = []

        for k, v in self.weight_dict.items():
            try:
                mean_mm.append(v[0].mean().item())
                var_mm.append(v[0].var().item())
            except:
                pass
        
        
        min_m = min(mean_mm)
        max_m = max(mean_mm)

        min_v = min(var_mm)
        max_v = max(var_mm)
        for k, v in  self.weight_dict.items():
            # if k in self.remove_index:
            #     continue
            try:
                mm_d['mean'][k] = (v[0].mean().item() - min_m)/(max_m - min_m)
                mm_d['var'][k] = (v[0].var().item() - min_v)/(max_v - min_v)
            except:
                pass

        self.all_dict.update(mm_d)
        self.all_dict.update(proxy_dict)
        
        metrics = [i for i in self.all_dict.keys()]
        all_files = []
        for i in range(1, 1+len(self.all_dict['mean'])):
            tmp = [str(i)]
            for metric in metrics:
                tmp.append(str(self.all_dict[metric][i]))
            all_files.append(tmp)
        path = f'./zeroshot/tmp/'
        # f'./zeroshot/{self.model_name}/{self.dataset_type}/'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # with open(os.path.join(path, 'results.csv'), 'w') as file:
        #     file.write('Layer_idx, ')
        #     for metric in metrics: 
        #         file.write(metric)
        #         file.write(', ')
        #     file.write('\n')
        #     for idx, file_ in enumerate(all_files):
        #         for el in file_: 
        #             file.write(el)
        #             file.write(', ')
        #         file.write('\n')
        self.make_input_format()
    def stabilize_model(self):
        # fix all layers except for conv2D and Linear
        # pass
        for _, p in self.model.net_g.named_parameters():
            p.requires_grad = False
        count = 0
        for layer in self.model.net_g.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                count += 1
                self.weight_dict[count] = []
                self.grads_dict[count] = []
                for _, p in layer.named_parameters():
                    p.requires_grad = True
    
    def get_dataset(self):
        lq_stack = []
        gt_stack = []
        for idx, img in enumerate(os.listdir(self.lq_pth)):
            tmp_lq = cv2.imread(os.path.join(self.lq_pth,img))
            tmp_gt = cv2.imread(os.path.join(self.gt_pth,img))
            lq_stack.append(tmp_lq)
            gt_stack.append(tmp_gt)

            if idx >= 1:
                break
            # np.stack
        lq_stack, gt_stack = self.simplify(lq_stack), self.simplify(gt_stack)
        return lq_stack, gt_stack
        # torch.tensor(np.stack(img2tensor(lq_stack))), torch.tensor(np.stack(img2tensor(gt_stack)))
    
    def get_static(self, ):
        # get a single batch of input, pass through model, and get their weights & gradients
        single_lq, single_gt = self.get_dataset()
        forward_params = []
        for name, param in self.model.net_g.named_parameters():
            if param.requires_grad:
                forward_params.append(param)
        self.optimizer_g = torch.optim.Adam([{'params': forward_params}],
                                                lr = self.lr, betas = (0.9, 0.999), eps=1e-08,)
        
        single_output = self.model.net_g(single_lq)
        loss = self.val_criterion(single_output, single_gt)
        loss.backward()
        self.get_weight_gradient()

    def get_weight_gradient(self):
        pass
        count = 0
        with torch.no_grad():
            for layer in self.model.net_g.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    count += 1
                    for _, p in layer.named_parameters():
                        self.weight_dict[count].append(p)
                        self.grads_dict[count].append(p.grad.data)
                        p.requires_grad = True
    
    def get_weight(self):
        pass
    def make_input_format(self):
        self.input_dict = {}
        # self.all_dict
        for i in range(1,227):
            tmp_store = []
            # self.input_dict[i] = []
            for zs_m in self.all_dict.keys():
                tmp_store.append(self.all_dict[zs_m][i])
            self.input_dict[i] = torch.tensor(tmp_store)
    def get_zeroshot(self):
        
        self.initialize_zs()
        single_lq, single_gt = self.get_dataset()
        for k, v in self.metric_dict.items():
            print(k)
            self.zs_dict[k] = v(self.model.net_g, single_lq,\
             single_gt, torch.nn.CrossEntropyLoss())
    
    def initialize_zs(self):
        self.metric_dict = {
            'grad_norm' : get_grad_norm_arr,
            'snip':compute_snip_per_weight , 
            'grasp': compute_grasp_per_weight, 
            'fisher': compute_fisher_per_weight, 
            'plain' : compute_plain_per_weight,
            'synflow': compute_synflow_per_weight 
        }
        for k, v in self.metric_dict.items():
            self.zs_dict[k] = {}
    
    def simplify(self, input):
        return torch.tensor(np.stack(img2tensor(input)).astype(np.float32))
    
    def train_network(self):
        pass
    
    def operation(self):
        for i, p in self.model.net_g.named_parameters():
            p.requires_grad = False
        random_lst = set()
        while True:
            random_num = random.randint(1,226)
            random_lst.add(random_num)
            if len(random_lst) == 40:
                break

        idx_num = 0
        for idx, layer in enumerate(self.model.net_g.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                idx_num += 1
                if idx_num in random_lst:
                    for _, p in layer.named_parameters():
                        p.requires_grad = True
        print(count)

