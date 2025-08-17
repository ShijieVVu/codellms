import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy as copy


class LoraLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int=4, dropout_p: float=0.0, scaling=16, test_mode: bool=False):
        super().__init__()
        self.base: nn.Linear = copy(base_layer)
        for p in self.base.parameters():
            p.requires_grad = False

        Cout, Cin = self.base.out_features, self.base.in_features
        self.B = nn.Parameter(torch.empty(Cout, r, dtype=self.base.weight.dtype))
        self.A = nn.Parameter(torch.empty(r, Cin, dtype=self.base.weight.dtype))
        self.r = r
        self.test_mode = test_mode
        self.dropout = nn.Dropout(dropout_p)
        self.scaling = scaling / r
        
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        if test_mode:
            nn.init.normal_(self.B, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.B)
    
    def forward(self, x):
        x = self.dropout(x)
        adjust = F.linear(F.linear(x, self.A), self.B)
        x = self.base(x) + adjust * self.scaling
        return x
    

def replace_linear_with_lora_(
        module: nn.Module, 
        r: int = 8,
        dropout_p: float=0.0,
        scaling: int=16,
        test_mode: bool=False, 
):
    for p in module.parameters():
        p.requires_grad = False
    
    def dfs(module, r, dropout_p, scaling, test_mode):
        for name, child in module.named_children():
            if any([key in name for key in ['wq', 'wk', 'wv', 'wo']]) and isinstance(child, nn.Linear):
                lora = LoraLinear(child, r, dropout_p, scaling, test_mode)
                setattr(module, name, lora)
            else:
                # leaf module has no children, non-leaf mode visited dfs
                dfs(child, r, dropout_p, scaling, test_mode)

    dfs(module, r, dropout_p, scaling, test_mode)

from llms.config import TrainArgs
from llms.llama import Transformer

args = TrainArgs()
transformer = Transformer(args)
print(transformer)
transformer.print_trainable_parameter()
replace_linear_with_lora_(transformer)
print(transformer)
transformer.print_trainable_parameter()