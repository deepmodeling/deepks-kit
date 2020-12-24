import torch
import torch.nn as nn 
import inspect
import math


def parse_actv_fn(code):
    if callable(code):
        return code
    assert type(code) is str
    lcode = code.lower()
    if lcode == 'sigmoid':
        return torch.sigmoid
    if lcode == 'tanh':
        return torch.tanh
    if lcode == 'relu':
        return torch.relu
    if lcode == 'softplus':
        return nn.Softplus()
    if lcode == 'gelu':
        return nn.functional.gelu
    if lcode == 'mygelu':
        return mygelu
    raise ValueError(f'{code} is not a valid activation function')


def mygelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


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
    
    def __init__(self, sizes, actv_fn=torch.relu, use_resnet=True, with_dt=False):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) 
                                     for in_f, out_f in zip(sizes, sizes[1:])])
        self.actv_fn = actv_fn
        self.use_resnet = use_resnet
        if with_dt:
            self.dts = nn.ParameterList([nn.Parameter(torch.normal(torch.ones(out_f), std=0.01)) 
                                         for out_f in sizes[1:]])
        else:
            self.dts = None
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            tmp = layer(x)
            if i < len(self.layers) - 1:
                tmp = self.actv_fn(tmp)
            if self.use_resnet and tmp.shape == x.shape:
                if self.dts is not None:
                    tmp = tmp * self.dts[i]
                x = x + tmp
            else:
                x = tmp
        return x
    

class CorrNet(nn.Module):

    @log_args('_init_args')
    def __init__(self, input_dim, hidden_sizes=(100,100,100), 
                 actv_fn='mygelu', use_resnet=True, 
                 input_shift=0, input_scale=1, output_scale=1):
        super().__init__()
        actv_fn = parse_actv_fn(actv_fn)
        layer_sizes = [input_dim, *hidden_sizes, 1]
        self.densenet = DenseNet(layer_sizes, actv_fn, use_resnet).double()
        self.linear = nn.Linear(input_dim, 1).double()
        self.input_shift = nn.Parameter(torch.tensor(input_shift, dtype=torch.float64).expand(input_dim).clone(), requires_grad=False)
        self.input_scale = nn.Parameter(torch.tensor(input_scale, dtype=torch.float64).expand(input_dim).clone(), requires_grad=False)
        self.output_scale = nn.Parameter(torch.tensor(output_scale, dtype=torch.float64), requires_grad=False)
        self.energy_const = nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
    
    def forward(self, x):
        # x: nframes x natom x nfeature
        x = (x - self.input_shift) / self.input_scale
        l = self.linear(x)
        y = self.densenet(x)
        y = y / self.output_scale + l
        e = y.sum(-2) + self.energy_const
        return e

    def set_normalization(self, shift=None, scale=None):
        dtype = self.input_scale.dtype
        if shift is not None:
            self.input_shift.data[:] = torch.tensor(shift, dtype=dtype)
        if scale is not None:
            self.input_scale.data[:] = torch.tensor(scale, dtype=dtype)

    def set_prefitting(self, weight, bias, trainable=False):
        dtype = self.linear.weight.dtype
        self.linear.weight.data[:] = torch.tensor(weight, dtype=dtype).reshape(-1)
        self.linear.bias.data[:] = torch.tensor(bias, dtype=dtype).reshape(-1)
        self.linear.requires_grad_(trainable)

    def set_energy_const(self, const):
        dtype = self.energy_const.dtype
        self.energy_const.data = torch.tensor(const, dtype=dtype).reshape([])

    def save_dict(self):
        dump_dict = {
            "state_dict": self.state_dict(),
            "init_args": self._init_args
        }
        return dump_dict

    def save(self, filename):
        torch.save(self.save_dict(), filename)
    
    @staticmethod
    def load_dict(checkpoint, strict=False):
        init_args = checkpoint["init_args"]
        if "layer_sizes" in init_args:
            layers = init_args.pop("layer_sizes")
            init_args["input_dim"] = layers[0]
            init_args["hidden_sizes"] = layers[1:-1]
        model = CorrNet(**init_args)
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        return model

    @staticmethod
    def load(filename, strict=False):
        checkpoint = torch.load(filename, map_location="cpu")
        return CorrNet.load_dict(checkpoint, strict=strict)
