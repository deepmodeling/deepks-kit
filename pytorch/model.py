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


class DenseNet(nn.Module):
    
    def __init__(self, sizes, actv_fn=torch.relu, use_resnet=True):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) 
                                     for in_f, out_f in zip(sizes, sizes[1:])])
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
        return x
    

class QCNet(nn.Module):

    @log_args('_init_args')
    def __init__(self, layer_sizes, actv_fn=torch.relu, use_resnet=False, input_shift=0, input_scale=1, output_scale=1):
        super().__init__()
        self.densenet = DenseNet(layer_sizes, actv_fn, use_resnet).double()
        self.input_shift = nn.Parameter(torch.tensor(input_shift, dtype=torch.float64).expand(layer_sizes[0]).clone(), requires_grad=False)
        self.input_scale = nn.Parameter(torch.tensor(input_scale, dtype=torch.float64).expand(layer_sizes[0]).clone(), requires_grad=False)
        self.output_scale = nn.Parameter(torch.tensor(output_scale, dtype=torch.float64), requires_grad=False)
    
    def forward(self, x):
        # x: nframes x natom x nfeature
        x = (x - self.input_shift) / self.input_scale
        y = self.densenet(x).sum(-2)
        y = y / self.output_scale
        return y

    def set_normalization(self, shift=None, scale=None):
        dtype = self.input_scale.dtype
        if shift is not None:
            self.input_shift.data = torch.tensor(shift, dtype=dtype)
        if scale is not None:
            self.input_scale.data = torch.tensor(scale, dtype=dtype)

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