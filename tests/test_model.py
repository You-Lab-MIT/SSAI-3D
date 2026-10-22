import numpy as np
import torch
import torch.nn as nn

from basicsr.models.archs import define_network
from ssai3d.config import load_opt
from ssai3d.restore import run_network
from ssai3d.surgeon import SurgeonMLP, _layers
from ssai3d.train import freeze_layers


def _network():
    opt = load_opt('finetune', device='cpu')
    return define_network(dict(opt['network_g']))


def _freeze_layer_v3(net, layer_idx):
    """Original implementation (v1.0), kept here as the reference."""
    for _, p in net.named_parameters():
        p.requires_grad = False
    li = [i[0] for i in layer_idx[:20]]
    count = 0
    idxs = []
    for idx, (_, p) in enumerate(net.named_parameters()):
        if p.numel() > 50000:
            count += 1
            idxs.append(idx)
        if count in li:
            p.requires_grad = True
    outputs = np.arange(idx - 20, idx).tolist()
    for idx, (_, p) in enumerate(net.named_parameters()):
        if idx in outputs or idx in idxs[:30] or idx in idxs[-30:]:
            p.requires_grad = True
    return net


def test_config_has_no_argparse_side_effects():
    opt = load_opt('finetune', is_train=True, experiments_root='/tmp/x', device='cpu')
    assert opt['num_gpu'] == 0 and opt['dist'] is False
    assert opt['path']['models'] == '/tmp/x/models'


def test_network_has_226_ranked_layers():
    assert len(_layers(_network())) == 226


def test_freeze_layers_matches_original():
    rng = np.random.default_rng(0)
    rank = [(int(i), 0.0) for i in rng.permutation(np.arange(1, 227))]
    a = [p.requires_grad for p in freeze_layers(_network(), rank).parameters()]
    b = [p.requires_grad for p in _freeze_layer_v3(_network(), rank).parameters()]
    assert a == b and 0 < sum(a) < len(a)


def test_surgeon_mlp_loads_bundled_weights():
    mlp = SurgeonMLP()
    rank = mlp.rank(torch.rand(10, 8))
    assert sorted(i for i, _ in rank) == list(range(1, 11))
    assert [s for _, s in rank] == sorted(s for _, s in rank)


def test_tiling_is_exact_for_pointwise_network():
    net = nn.Conv2d(3, 3, 1)
    x = torch.rand(1, 3, 70, 50)
    with torch.no_grad():
        np.testing.assert_allclose(run_network(net, x, tile=32, overlap=8),
                                   net(x), atol=1e-6)
