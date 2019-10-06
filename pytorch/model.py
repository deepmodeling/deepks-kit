import torch
import torch.nn as nn 
import inspect


def log_args(name):
    def decorator(func):
        def warpper(self, *args, **kwargs):
            args_dict = inspect.getcallargs(func, self, *args, **kwargs)
            del args_dict['self']
            setattr(self, name, args_dict)
            func(self, *args, **kwargs)
        return warpper
    return decorator


class ConvNet1d(nn.Module):
    
    def __init__(self, channel_sizes, kernel_size, actv_fn=torch.relu, use_resnet=True):
        super().__init__()
        padding = (kernel_size-1) // 2
        self.layers = nn.ModuleList([nn.Conv1d(in_c, out_c, kernel_size, padding=padding) 
                                     for in_c, out_c in zip(channel_sizes, channel_sizes[1:])])
        self.actv_fn = actv_fn
        self.use_resnet = use_resnet
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            tmp = layer(x)
            if i < len(self.layers) - 1:
                tmp = self.actv_fn(tmp)
            if self.use_resnet and tmp.shape == x.shape:
                x = x + tmp
            else:
                x = tmp
        return x # N x 1(n_channel_out) x n_orbits
    

class QCNet(nn.Module):

    @log_args('_init_args')
    def __init__(self, channel_sizes, kernel_size=5, actv_fn=torch.relu, use_resnet=False, input_scale=1, output_scale=1):
        super().__init__()
        self.convnet = ConvNet1d(channel_sizes, kernel_size, actv_fn, use_resnet)
        self.input_scale = input_scale
        self.output_scale = output_scale
    
    def forward(self, x):
        x = x / self.input_scale
        y = self.convnet(x).sum(-1)
        y = y / self.output_scale
        return y

    def save(self, filename):
        dump_dict = {
            "state_dict": self.state_dict(),
            "init_args": self._init_args
        }
        torch.save(dump_dict, filename)
    
    @staticmethod
    def load(filename):
        checkpoint = torch.load(filename, map_location="cpu")
        model = QCNet(**checkpoint["init_args"])
        model.load_state_dict(checkpoint['state_dict'])
        return model
