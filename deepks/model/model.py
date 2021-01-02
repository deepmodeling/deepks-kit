import math
import inspect
import torch
import torch.nn as nn 
from torch.nn import functional as F
from deepks.utils import load_basis, get_shell_sec

SCALE_EPS = 1e-8


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
    if lcode == 'silu':
        return nn.SiLU()
    if lcode == 'gelu':
        return F.gelu
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
            self.dts = nn.ParameterList(
                [nn.Parameter(torch.normal(torch.ones(out_f), std=0.01)) 
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


def make_embedder(type, shell_sec, **kwargs):
    ltype = type.lower()
    if ltype in ("trace", "sum"):
        EmbdCls = TraceEmbedding
    elif ltype in ("thermal", "softmax"):
        EmbdCls = ThermalEmbedding
    else:
        raise ValueError(f'{type} is not a valid embedding type')
    embedder = EmbdCls(shell_sec, **kwargs)
    return embedder


class TraceEmbedding(nn.Module):

    def __init__(self, shell_sec):
        super().__init__()
        self.shell_sec = shell_sec
        self.ndesc = len(shell_sec)
    
    def forward(self, x):
        x_shells = x.split(self.shell_sec, dim=-1)
        tr_shells = [sx.sum(-1, keepdim=True) for sx in x_shells]
        return torch.cat(tr_shells, dim=-1)
    

class ThermalEmbedding(nn.Module):

    def __init__(self, shell_sec, embd_sizes=None, init_beta=1., momentum=None):
        super().__init__()
        input_dim = sum(shell_sec)
        self.shell_sec = shell_sec
        if embd_sizes is None:
            embd_sizes = shell_sec
        if isinstance(embd_sizes, int):
            embd_sizes = [embd_sizes] * len(shell_sec)
        assert len(embd_sizes) == len(shell_sec)
        self.embd_sizes = embd_sizes
        self.ndesc = sum(embd_sizes)
        self.beta_shells = nn.ParameterList(
            [nn.Parameter(torch.linspace(init_beta, -init_beta, ne))
                for ne in embd_sizes])
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(len(shell_sec)))
        self.register_buffer('running_var', torch.ones(len(shell_sec)))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x_shells = x.split(self.shell_sec, dim=-1)
        if self.training:
            self.update_running_stats(x_shells)
        nx_shells = [(sx - m) / (v.sqrt() + SCALE_EPS) 
            for sx, m, v in zip(x_shells, self.running_mean, self.running_var)]
        desc_shells = [
            torch.sum(sx.unsqueeze(-1) 
                * F.softmax(-sbeta * snx.unsqueeze(-1), dim=-2), dim=-2)
            for sx, snx, sbeta in zip(x_shells, nx_shells, self.beta_shells)
        ]
        return torch.cat(desc_shells, dim=-1)

    def update_running_stats(self, x_shells):
        self.num_batches_tracked += 1
        exp_factor = 1. - 1. / float(self.num_batches_tracked)
        if self.momentum is not None:
            exp_factor = max(exp_factor, self.momentum)
        with torch.no_grad():
            batch_mean = torch.stack([sx.detach().mean() for sx in x_shells])
            batch_var = torch.stack([sx.detach().var() for sx in x_shells])
            self.running_mean[:] = exp_factor * self.running_mean + (1-exp_factor) * batch_mean
            self.running_var[:] = exp_factor * self.running_var + (1-exp_factor) * batch_var
        
    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()



class CorrNet(nn.Module):

    @log_args('_init_args')
    def __init__(self, input_dim, hidden_sizes=(100,100,100), 
                 actv_fn='gelu', use_resnet=True, 
                 embedding=None, proj_basis=None,
                 input_shift=0, input_scale=1, output_scale=1):
        super().__init__()
        actv_fn = parse_actv_fn(actv_fn)
        # basis info
        self._pbas = load_basis(proj_basis)
        self._init_args["proj_basis"] = self._pbas
        self.shell_sec = None
        # linear fitting
        self.linear = nn.Linear(input_dim, 1).double()
        # embedding net
        ndesc = input_dim
        self.embedder = None
        if embedding is not None:
            if isinstance(embedding, str):
                embedding = {"type": embedding}
            assert isinstance(embedding, dict)
            self.shell_sec = get_shell_sec(self._pbas)
            self.embedder = make_embedder(**embedding, shell_sec=self.shell_sec).double()
            self.linear.requires_grad_(False) # make sure it is symmetric
            ndesc = self.embedder.ndesc
        # fitting net
        layer_sizes = [ndesc, *hidden_sizes, 1]
        self.densenet = DenseNet(layer_sizes, actv_fn, use_resnet).double()
        # scaling part
        self.input_shift = nn.Parameter(
            torch.tensor(input_shift, dtype=torch.float64).expand(input_dim).clone(), 
            requires_grad=False)
        self.input_scale = nn.Parameter(
            torch.tensor(input_scale, dtype=torch.float64).expand(input_dim).clone(), 
            requires_grad=False)
        self.output_scale = nn.Parameter(
            torch.tensor(output_scale, dtype=torch.float64), 
            requires_grad=False)
        self.energy_const = nn.Parameter(
            torch.tensor(0, dtype=torch.float64), 
            requires_grad=False)
    
    def forward(self, x):
        # x: nframes x natom x nfeature
        x = (x - self.input_shift) / (self.input_scale + SCALE_EPS)
        l = self.linear(x)
        if self.embedder is not None:
            x = self.embedder(x)
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

    def save_dict(self, **extra_info):
        dump_dict = {
            "state_dict": self.state_dict(),
            "init_args": self._init_args,
            "extra_info": extra_info
        }
        return dump_dict

    def save(self, filename, **extra_info):
        torch.save(self.save_dict(**extra_info), filename)
    
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
