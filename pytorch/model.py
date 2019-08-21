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
            self.dts = nn.ParameterList([nn.Parameter(torch.normal(torch.ones(out_f), std=0.01)) for out_f in sizes[1:]])
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
    
    The descriptor is given by 
    $ d_{i,I,f,l} = (1/N_orbit_j) * (1/N_proj) * \sum_{j, a} <i|f|R_I,a><R_I,a|f|j> g_l(e_j) $.

    Args:
        n_neuron: the shape of layers used in the filter network, input size not included
    
    Shape:
        input:
            mo_i: n_frame x n_orbit_i x n_atom x n_proj
            e_i:  n_frame x n_orbit_i
            mo_j: n_frame x n_orbit_j x n_atom x n_proj
            e_j:  n_frame x n_orbit_j
        output:
            d: n_frame x n_atom x n_orbit_i x n_filter
    """
    def __init__(self, n_neuron):
        super().__init__()
        self.filter = DenseNet([1] + n_neuron, use_resnet=True)
    
    def forward(self, mo_i, e_i, mo_j, e_j):
        N, no_i, na, np = mo_i.shape
        N, no_j, na, np = mo_j.shape
        g_j = self.filter(e_j.unsqueeze(-1)) # n_frame x n_orbit_j x n_filter (nov)
        u_j = torch.einsum("nov,noapf->napfv", g_j, mo_j) / no_j # n_frame x n_atom x n_operator x n_proj x n_filter
        d = torch.einsum("noapf,napfv->noafv", mo_i, u_j) / np # n_frame x n_orbit_i x n_atom x n_operator x n_filter
        return d.flatten(-2)


class ShellDescriptor(nn.Module):
    r"""module to calculate descriptor for each shell from (projected) MO and (baseline) energy
    
    For each shell the descriptor is given by 
    $ d_{i,I,f,l} = (1 / N_orbit_j) * (1 / N_shell) * \sum_{j, a} <i|f|R_I,a><R_I,a|f|j> g_l(e_j) $. 

    Args:
        n_neuron: the shape of layers used in the filter network, input size not included
        shell_sections: a list of the number of orbitals for each shell to be summed up
    
    Shape:
        input:
            mo_i: n_frame x n_orbit_i x n_atom x n_proj x n_operator (noapf)
            e_i:  n_frame x n_orbit_i
            mo_j: n_frame x n_orbit_j x n_atom x n_proj x n_operator (noapf)
            e_j:  n_frame x n_orbit_j
        output:
            d: n_frame x n_atom x n_orbit_i x (n_filter x n_shell)
    """
    def __init__(self, n_neuron, shell_sections):
        super().__init__()
        self.filter = DenseNet([1] + n_neuron)
        self.sections = shell_sections
        # self.layer_norm = nn.LayerNorm([n_neuron[-1], len(shell_sections)])
    
    def forward(self, mo_i, e_i, mo_j, e_j):
        N, no_i, na, np, nf = mo_i.shape
        N, no_j, na, np, nf = mo_j.shape
        assert sum(self.sections) == np
        g_j = self.filter(e_j.unsqueeze(-1)) # n_frame x n_orbit_j x n_filter (nov)
        u_j = torch.einsum("nov,noapf->napfv", g_j, mo_j) / no_j # n_frame x n_atom x n_operator x n_proj x n_filter
        u_j_list = torch.split(u_j, self.sections, dim=-3) # [n_frame x n_atom x n_operator x n_ao_in_shell x n_filter] list
        mo_i_list = torch.split(mo_i, self.sections, dim=-2) # [n_frame x n_orbit x n_atom x n_ao_in_shell x n_operator] list
        d_list = [torch.einsum("noapf,napfv->noafv", _mo_i, _u_j) / _np # n_frame x n_orbit_i x n_atom x n_operator x n_filter
                    for _mo_i, _u_j, _np in zip(mo_i_list, u_j_list, self.sections)] 
        d = torch.stack(d_list, dim=-1) # n_frame x n_orbit_i x n_atom x n_operator x n_filter x n_shell
        # d = self.layer_norm(d) # layer normalization
        return d.flatten(-3) # n_frame x n_orbit_i x n_atom x (n_operator x n_filter x n_shell)


class QCNet(nn.Module):
    """our quantum chemistry model

    The model is given by $ E_i^corr = \sum_I f( d_i^occ(R_I), d_i^vir(R_I) ) $ 
    and $ E^corr = \sum_i E_i^corr $,
    where $d$ is calculated by `Descriptor` module.

    Args:
        n_neuron_d: the shape of layers used in descriptor's network
        n_neuron_e: the shape of layers used in fitting network $f$
        shell_sections: a shell list to split projected orbits into and do summation separately
        e_stat: (e_avg, e_stat), if given, would scale the input energy accordingly
            Default: None
        use_resnet: whether to use resnet structure in fitting network
            Default: False

    Shape:
        input:
            mo_occ: n_frame x n_occ x n_atom x n_proj x n_operator
            e_occ:  n_frame x n_occ
            mo_vir: n_frame x n_vir x n_atom x n_proj x n_operator
            e_vir:  n_frame x n_vir
        output:
            e_corr: n_frame
    """
    @log_args('_init_args')
    def __init__(self, n_neuron_d, n_neuron_e, shell_sections, n_operator=1, e_stat=None, c_stat=None, use_resnet=False):
        super().__init__()
        self._nop = n_operator
        if shell_sections is None:
            nd = n_operator * n_neuron_d[-1]
            self.dnet_occ = Descriptor(n_neuron_d)
            self.dnet_vir = Descriptor(n_neuron_d)
        else:
            nd = n_operator * n_neuron_d[-1] * len(shell_sections)
            self.dnet_occ = ShellDescriptor(n_neuron_d, shell_sections)
            self.dnet_vir = ShellDescriptor(n_neuron_d, shell_sections)
        self.enet = DenseNet([2 * nd] + n_neuron_e, use_resnet=use_resnet, with_dt=True)
        self.final_layer = nn.Linear(n_neuron_e[-1], 1, bias=False)
        if e_stat is not None:
            self.scale_e = True
            e_avg, e_std = e_stat
            self.e_avg = nn.Parameter(torch.tensor(e_avg))
            self.e_std = nn.Parameter(torch.tensor(e_std))
        else:
            self.scale_e = False
        if c_stat is not None:
            self.scale_c = True
            c_avg, c_std = c_stat
            self.c_avg = nn.Parameter(torch.tensor(c_avg))#, requires_grad=False)
            self.c_std = nn.Parameter(torch.tensor(c_std))#, requires_grad=False)
        else:
            self.scale_c = False
    
    def forward(self, mo_occ, mo_vir, e_occ, e_vir):
        if self.scale_e:
            e_occ = (e_occ - self.e_avg) / self.e_std
            e_vir = (e_vir - self.e_avg) / self.e_std
        if self.scale_c:
            mo_occ = (mo_occ - self.c_avg) / self.c_std
            mo_vir = (mo_vir - self.c_avg) / self.c_std
        mo_occ = mo_occ[:,:,:,:,:self._nop]
        mo_vir = mo_vir[:,:,:,:,:self._nop]
        d_occ = self.dnet_occ(mo_occ, e_occ, mo_occ, e_occ) # n_frame x n_occ x n_atom x n_des
        d_vir = self.dnet_vir(mo_occ, e_occ, mo_vir, e_vir) # n_frame x n_occ x n_atom x n_des
        d_all = torch.cat([d_occ, d_vir], dim=-1) # n_frame x n_occ x n_atom x 2 n_d
        d_all = torch.tanh(d_all)
        e_all = self.final_layer(self.enet(d_all)) # n_frame x n_occ x n_atom x 1
        e_corr = torch.sum(e_all, dim=[1,2,3])
        return e_corr

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
