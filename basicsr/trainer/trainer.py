import torch.nn as nn
import torch

class SurgeonTrainer(nn.Module):
    def __init__(self, input_dims = 8) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.mlp = nn.Sequential(
            nn.Linear(input_dims , 64), 
            nn.GELU(),
            nn.Linear(64, 128),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(64,1),)
        
    def forward(self, x1, x2):
        x1 = self.mlp(x1)
        x2 = self.mlp(x2)
        return x1, x2
    
    def forward_all(self, input_layers):
        tmp_dict = []
        for layer_idx, zs_value in input_layers.items():
            assert zs_value.shape[0] == self.input_dims
            zs_rank = self.mlp(zs_value)
            zs_rank = -1 * zs_rank.detach().cpu().item()
            tmp_dict.append((layer_idx, zs_rank))
        tmp_dict = sorted(tmp_dict, key = lambda x : x[1])
        return tmp_dict