# from utils.mobilenetv2 import MobileNetV2
# from utils.get_layer import get_whole_layers
import torch 
from typing import List
import numpy as np
def freeze_layer(net, layer_id):
    for l in net.parameters():
        l.requires_grad = False
    for id, layer in enumerate(net.parameters()):
        if id == layer_id or id == 158 or id ==159:
            layer.requires_grad = True
    return net

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    r"""
    Computes the precision@k for the specified values of k
    """
    maxk = min(max(topk), output.shape[1])
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if k <= output.shape[1]:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(torch.zeros(1).cuda() - 1.)
    return res

def freeze_layer_v2(net, layer,linear_layer):
    for id, layer_ in enumerate(net.named_parameters()):
        name , param = layer_
        if name.startswith(layer) or name.startswith(linear_layer): 
            param.requires_grad = True
        else: 
            param.requires_grad = False
    return net

def freeze_layer_v3(neural_network, layer_idx):
    for _, p in neural_network.named_parameters():
        p.requires_grad = False

    li = [i[0] for i in layer_idx[:20]]
    count = 0
    idxs = []
    for idx, (_, p) in enumerate(neural_network.named_parameters()):
        if p.numel()> 50000:
            count += 1
            idxs.append(idx)
        if count in li:
            p.requires_grad = True
    outputs = np.arange(idx-20,idx).tolist()
    for idx, (_, p) in enumerate(neural_network.named_parameters()):
        if idx in outputs or idx in idxs[:30] or idx in idxs[-30:]:
            p.requires_grad = True
    return neural_network


def get_weights(net, layer):
    weight_dic = {}
    for name, parameters in net.named_parameters():
        if name.startswith(layer):
            weight_dic[name] = parameters
    return weight_dic
    

if __name__ == '__main__':
    net = MobileNetV2()
    # layers = get_whole_layers(net)
    # freeze_layer_v2(net, 'features.8')
    weight = get_weights(net, 'features.8')
    print(weight)
    