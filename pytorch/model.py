import torch
import torch.nn as nn 


class DenseNet(nn.Module):
    r"""module to create a dense network with given size, activation function and optional resnet structure

    Args:
        sizes: the shape of each layers, including the input size at begining
        actv_fn: activation function used after each layer's linear transformation
            Default: `torch.tanh`
        use_resnet: whether to use resnet structure between layers with same size or doubled size
            Default: `True`
        with_dt: whether to multiply a timestep in resnet sturcture, only effective when `use_resnet=True`
            Default: `False`
    """
    def __init__(self, sizes, actv_fn=torch.tanh, use_resnet=True, with_dt=False):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.actv_fn = actv_fn
        self.use_resnet = use_resnet
        if with_dt:
            self.dts = nn.ParameterList([nn.Parameter(torch.normal(torch.ones(out_f), std=0.1)) for out_f in sizes[1:]])
        else:
            self.dts = None

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            tmp = self.actv_fn(layer(x))
            if self.use_resnet and (tmp.shape == x.shape or tmp.shape[-1] == 2*x.shape[-1]):
                if self.dts is not None:
                    tmp = tmp * self.dts[i]
                if tmp.shape[-1] == 2*x.shape[-1]:
                    x = torch.cat([x, x], dim=-1)
                x = x + tmp
            else:
                x = tmp
        return x
    

class Descriptor(nn.Module):
    r"""module to calculate descriptor from given (projected) molecular orbitals and (baseline) energy
    
    The descriptor is given by $ d_I = (1 / N_orbit) * (1 / N_proj) 
                                        * \sum_{i, j, a} g(e_i) <i|R_I,a><R_I,a|j>g^T(e_j) 
                                        = (1 / N_proj) * u \dot u^T $.

    Args:
        n_neuron: the shape of layers used in the filter network, input size not included.
    
    Shape:
        input:
            mo: n_frame x n_orbit x n_atom x n_proj
            e:  n_frame x n_orbit
        output:
            d: n_frame x n_atom x (n_filter x n_filter)
    """
    def __init__(self, n_neuron):
        super().__init__()
        self.filter = DenseNet([1] + n_neuron)
    
    def forward(self, mo, e):
        nf, no, na, np = mo.shape
        g = self.filter(e.unsqueeze(-1)) # n_frame x n_orbit x n_filter
        # u = (g.transpose(1,2) @ mo.reshape(nf, no, -1)) # n_frame x n_filter x (n_atom x n_proj)
        # u = u.reshape(nf, -1, na, np).transpose(1,2) # n_frame x n_atom x n_filter x n_proj
        u = torch.einsum("nof,noap->nafp", g, mo) / no # n_frame x n_atom x n_filter x n_proj
        d = u @ u.transpose(-2,-1) / np # n_frame x n_atom x n_filter x n_filter
        return d.flatten(-2)


class QCNet(nn.Module):
    """our quantum chemistry model

    The model is given by $ E_corr = \sum_I f( d^occ(R_I), d^vir(R_I) ) $ and $d$ is calculated by `Descriptor` module.

    Args:
        n_neuron_filter: the shape of layers used in descriptor's filter
        n_neuron_fit: the shape of layers used in fitting network $f$
        e_stat: (e_avg, e_stat), if given, would scale the input energy accordingly
            Default: None
        use_resnet: whether to use resnet structure in fitting network

    Shape:
        input:
            mo_occ: n_frame x n_occ x n_atom x n_proj
            e_occ:  n_frame x n_occ
            mo_vir: n_frame x n_vir x n_atom x n_proj
            e_vir:  n_frame x n_vir
        output:
            e_corr: n_frame
    """
    def __init__(self, n_neuron_filter, n_neuron_fit, e_stat=None, use_resnet=False):
        super().__init__()
        self.dnet_occ = Descriptor(n_neuron_filter)
        self.dnet_vir = Descriptor(n_neuron_filter)
        self.fitnet = DenseNet([2*n_neuron_filter[-1]**2] + n_neuron_fit, use_resnet=use_resnet, with_dt=True)
        self.final_layer = nn.Linear(n_neuron_fit[-1], 1, bias=False)
        if e_stat is not None:
            self.scale_e = True
            e_avg, e_std = e_stat
            self.e_avg = nn.Parameter(torch.tensor(e_avg))
            self.e_std = nn.Parameter(torch.tensor(e_std))
        else:
            self.scale_e = False
    
    def forward(self, mo_occ, mo_vir, e_occ, e_vir):
        if self.scale_e:
            e_occ = (e_occ - self.e_avg) / self.e_std
            e_vir = (e_vir - self.e_avg) / self.e_std
        d_occ = self.dnet_occ(mo_occ, e_occ) # n_frame x n_atom x n_filter^2
        d_vir = self.dnet_vir(mo_vir, e_vir) # n_frame x n_atom x n_filter^2
        d_atom = torch.cat([d_occ, d_vir], dim=-1) # n_frame x n_atom x (2 x n_filter^2)
        e_atom = self.final_layer(self.fitnet(d_atom)) # n_frame x n_atom x 1
        e_corr = torch.sum(e_atom, dim=[1,2])
        return e_corr

