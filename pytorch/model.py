import torch
import torch.nn as nn 


class DenseNet(nn.Module):
    def __init__(self, sizes, actv_fn=torch.tanh, use_resnet=True):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.actv_fn = actv_fn
        self.use_resnet = use_resnet

    def forward(self, x):
        for layer in self.layers:
            tmp = self.actv_fn(layer(x))
            if self.use_resnet and tmp.shape[-1] % x.shape[-1] == 0:
                x = x + tmp
            else:
                x = tmp
        return x
    

class QCNet(nn.Module):
    pass