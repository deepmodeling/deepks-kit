import math
import inspect
import numpy as np
import torch
import torch.nn as nn 
from torch.nn import functional as F
from deepks.utils import load_basis, get_shell_sec
from deepks.utils import load_elem_table, save_elem_table

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
        return F.softplus
    if lcode == 'silu':
        return F.silu
    if lcode == 'gelu':
        return F.gelu
    if lcode == 'mygelu':
        return mygelu
    raise ValueError(f'{code} is not a valid activation function')


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


def make_shell_mask(shell_sec):
    lsize = len(shell_sec)
    msize = max(shell_sec)
    mask = torch.zeros(lsize, msize, dtype=bool)
    for l, m in enumerate(shell_sec):
        mask[l, :m] = 1
    return mask


def pad_lastdim(sequences, padding_value=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    front_dims = max_size[:-1]
    max_len = max([s.size(-1) for s in sequences])
    out_dims = front_dims + (len(sequences), max_len)
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(-1)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[..., i, :length] = tensor
    return out_tensor


def pad_masked(tensor, mask, padding_value=0):
    # equiv to pad_lastdim(tensor.split(shell_sec, dim=-1))
    assert tensor.shape[-1] == mask.sum()
    new_shape = tensor.shape[:-1] + mask.shape
    return tensor.new_full(new_shape, padding_value).masked_scatter_(mask, tensor) 


def unpad_lastdim(padded, length_list):
    # inverse of pad_lastdim
    return [padded[...,i,:length] for i, length in enumerate(length_list)]


def unpad_masked(padded, mask):
    # equiv to torch.cat(unpad_lastdim(padded, shell_sec), dim=-1)
    new_shape = padded.shape[:-mask.ndim] + (mask.sum(),)
    return torch.masked_select(padded, mask).reshape(new_shape)


def masked_softmax(input, mask, dim=-1):
    exps = torch.exp(input - input.max(dim=dim, keepdim=True)[0])
    mexps = exps * mask.to(exps)
    msums = mexps.sum(dim=dim, keepdim=True).clamp(1e-10)
    return mexps / msums


class DenseNet(nn.Module):
    
    def __init__(self, sizes, actv_fn=torch.relu, 
                 use_resnet=True, with_dt=False, layer_norm=False):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])
        ])
        self.ln_layers = nn.ModuleList([
            nn.LayerNorm(in_f, elementwise_affine=(layer_norm!="simple"))
            for in_f in sizes[:-1]
        ]) if layer_norm else [None] * (len(sizes)-1)
        self.actv_fn = actv_fn
        self.use_resnet = use_resnet
        if with_dt:
            self.dts = nn.ParameterList(
                [nn.Parameter(torch.normal(torch.ones(out_f), std=0.01)) 
                    for out_f in sizes[1:]])
        else:
            self.dts = None
    
    def forward(self, x):
        for i, (layer, ln_layer) in enumerate(zip(self.layers, self.ln_layers)):
            tmp = x
            if ln_layer is not None:
                tmp = ln_layer(tmp)
            tmp = layer(tmp)
            if i < len(self.layers) - 1:
                tmp = self.actv_fn(tmp)
            if self.use_resnet and layer.in_features == layer.out_features:
                if self.dts is not None:
                    tmp = tmp * self.dts[i]
                x = x + tmp
            else:
                x = tmp
        return x


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

    def __init__(self, shell_sec, embd_sizes=None, init_beta=5., 
                 momentum=None, max_memory=1000):
        super().__init__()
        self.shell_sec = shell_sec
        self.register_buffer("shell_mask", make_shell_mask(shell_sec), False)# shape: [l, m]
        if embd_sizes is None:
            embd_sizes = shell_sec
        if isinstance(embd_sizes, int):
            embd_sizes = [embd_sizes] * len(shell_sec)
        assert len(embd_sizes) == len(shell_sec)
        self.embd_sizes = embd_sizes
        self.register_buffer("embd_mask", make_shell_mask(embd_sizes), False)
        self.ndesc = sum(embd_sizes)
        self.beta = nn.Parameter( # shape: [l, p], padded
            pad_lastdim([torch.linspace(init_beta, -init_beta, ne) 
                            for ne in embd_sizes]))
        self.momentum = momentum
        self.max_memory = max_memory
        self.register_buffer('running_mean', torch.zeros(len(shell_sec)))
        self.register_buffer('running_var', torch.ones(len(shell_sec)))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x_padded = pad_masked(x, self.shell_mask, 0.) # shape: [n, a, l, m]
        if self.training:
            self.update_running_stats(x_padded)
        nx_padded = ((x_padded - self.running_mean.unsqueeze(-1)) 
                    / (self.running_var.sqrt().unsqueeze(-1) + SCALE_EPS)
                    * self.shell_mask.to(x_padded))
        weight = masked_softmax(
            torch.einsum("...lm,lp->...lmp", nx_padded, -self.beta),
            self.shell_mask.unsqueeze(-1), dim=-2)
        desc_padded = torch.einsum("...m,...mp->...p", x_padded, weight)
        return unpad_masked(desc_padded, self.embd_mask)

    def update_running_stats(self, x_padded):
        self.num_batches_tracked += 1
        if self.momentum is None and self.num_batches_tracked > self.max_memory:
            return # stop update after 1000 batches, so the scaling becomes a fixed parameter
        exp_factor = 1. - 1. / float(self.num_batches_tracked)
        if self.momentum is not None:
            exp_factor = max(exp_factor, self.momentum)
        with torch.no_grad():
            fmask = self.shell_mask.to(x_padded)
            pad_portion = fmask.mean(-1)
            x_masked = x_padded * fmask # make sure padded part is zero
            reduced_dim = (*range(x_masked.ndim-2), -1)
            batch_mean = x_masked.mean(reduced_dim) / pad_portion
            batch_var = x_masked.var(reduced_dim) / pad_portion
            self.running_mean[:] = exp_factor * self.running_mean + (1-exp_factor) * batch_mean
            self.running_var[:] = exp_factor * self.running_var + (1-exp_factor) * batch_var
        
    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()


class CorrNet(nn.Module):

    @log_args('_init_args')
    def __init__(self, input_dim, hidden_sizes=(100,100,100), 
                 actv_fn='gelu', use_resnet=True, layer_norm=False,
                 embedding=None, proj_basis=None, elem_table=None,
                 input_shift=0, input_scale=1, output_scale=1):
        super().__init__()
        actv_fn = parse_actv_fn(actv_fn)
        self.input_dim = input_dim
        # basis info
        self._pbas = load_basis(proj_basis)
        self._init_args["proj_basis"] = self._pbas
        self.shell_sec = None
        # elem const
        if isinstance(elem_table, str):
            elem_table = load_elem_table(elem_table)
            self._init_args["elem_table"] = elem_table
        self.elem_table = elem_table
        self.elem_dict = None if elem_table is None else dict(zip(*elem_table))
        # linear fitting
        self.linear = nn.Linear(input_dim, 1).double()
        self.elem_table=elem_table
        # embedding net
        ndesc = input_dim
        self.embedder = None
        if embedding is not None:
            if isinstance(embedding, str):
                embedding = {"type": embedding}
            assert isinstance(embedding, dict)
            raw_shell_sec = get_shell_sec(self._pbas)
            self.shell_sec = raw_shell_sec * (input_dim // sum(raw_shell_sec))
            assert sum(self.shell_sec) == input_dim
            self.embedder = make_embedder(**embedding, shell_sec=self.shell_sec).double()
            self.linear.requires_grad_(False) # make sure it is symmetric
            ndesc = self.embedder.ndesc
        # fitting net
        layer_sizes = [ndesc, *hidden_sizes, 1]
        self.densenet = DenseNet(
            sizes=layer_sizes, 
            actv_fn=actv_fn, 
            use_resnet=use_resnet,
            layer_norm=layer_norm).double()
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
    
    def get_elem_const(self, elems):
        if self.elem_dict is None:
            return 0.
        return sum(self.elem_dict[ee] for ee in elems)

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

    def compile(self, set_eval=True, **kwargs):
        old_mode = self.training
        if set_eval:
            self.eval()
        smodel = torch.jit.trace(
            self.forward, 
            torch.empty((2, 2, self.input_dim)),
            **kwargs)
        self.train(old_mode)
        return smodel

    def compile_save(self, filename, **kwargs):
        torch.jit.save(self.compile(**kwargs), filename)
        if self.elem_table is not None:
            save_elem_table(filename+".elemtab", self.elem_table)
    
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
        try:
            return torch.jit.load(filename)
        except RuntimeError:
            checkpoint = torch.load(filename, map_location="cpu")
            return CorrNet.load_dict(checkpoint, strict=strict)
