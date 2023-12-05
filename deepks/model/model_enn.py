
import torch
import torch.nn as nn

import e3nn.o3
import e3nn.nn
from e3nn.util.jit import compile_mode

from deepks.model.model import log_args


class CorrNet(nn.Module):

    @log_args('_init_args')
    def __init__(self,
                 irreps_in, irreps_hidden=None, irreps_tp=None, irreps_out='0e',
                 tp_features=10,
                 actv_type="gate",
                 actv_scalars={"e": "silu", "o": "tanh"},
                 actv_gates={"e": "relu", "o": "tanh"},
                 mlp_layers=3, use_resnet=True,
                 prefit=False, output_scale=1., input_shift=0.
                 ):

        if actv_type not in ("gate", "norm"):
            raise NotImplementedError("activation function type can only be gate or norm")

        super().__init__()

        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.irreps_out = e3nn.o3.Irreps(irreps_out)
        if irreps_tp is None:
            self.irreps_tp = str(e3nn.o3.Irreps([(tp_features, ir) for _, ir in self.irreps_in]))

        self.n_scalar_in = self.irreps_in.count(e3nn.o3.Irrep('0e'))

        if irreps_hidden is None:
            mlp_irreps = [irreps_in] + (mlp_layers-1)*[irreps_in] + [self.irreps_tp]
        elif irreps_hidden == 'tp':
            mlp_irreps = [irreps_in] + (mlp_layers-1)*[self.irreps_tp] + [self.irreps_tp]
        elif isinstance(irreps_hidden, str):
            mlp_irreps = [irreps_in] + (mlp_layers-1)*[irreps_hidden] + [self.irreps_tp]
        elif isinstance(irreps_hidden, list):
            mlp_irreps = [irreps_in] + irreps_hidden + [self.irreps_tp]
        else:
            raise RuntimeError('Fail to generate mlp irreps')

        self.prefit = prefit
        if prefit:
            self.torch_linear = torch.nn.Linear(self.n_scalar_in, 1)

        self.mlp = MLP(irreps=mlp_irreps,
                       actv_type=actv_type,
                       actv_scalars=actv_scalars,
                       actv_gates=actv_gates,
                       num_layers=mlp_layers,
                       resnet=use_resnet)
        self.tp_out = e3nn.o3.FullyConnectedTensorProduct(irreps_in1=self.irreps_tp,
                                                          irreps_in2=self.irreps_tp,
                                                          irreps_out=self.irreps_out)

        shift = torch.zeros(self.irreps_in.dim, dtype=torch.float64)
        shift[:self.n_scalar_in] = torch.tensor(input_shift, dtype=torch.float64).expand(self.n_scalar_in)
        self.input_shift = nn.Parameter(
            shift.clone(),
            requires_grad=False)
        self.output_scale = nn.Parameter(
            torch.tensor(output_scale, dtype=torch.float64),
            requires_grad=False)

    def set_prefitting(self, weight, bias, trainable=False):

        if self.prefit:
            dtype = self.torch_linear.weight.dtype
            self.torch_linear.weight.data[:] = torch.as_tensor(weight, dtype=dtype).reshape(-1)
            self.torch_linear.bias.data[:] = torch.as_tensor(bias, dtype=dtype).reshape(-1)
            self.torch_linear.requires_grad_(trainable)

    def set_preshift(self, shift):

        dtype = self.input_shift.dtype
        self.input_shift.data[..., :self.n_scalar_in] = torch.as_tensor(shift, dtype=dtype)

    def forward(self, x):

        x = x - self.input_shift
        if self.prefit:
            y_fit = self.torch_linear(x[..., :self.n_scalar_in])
        x = self.mlp(x)
        y = self.tp_out(x, x) / self.output_scale
        if self.prefit:
            y += y_fit

        return y

    def save(self, filename):
        torch.save(
            {"state_dict": self.state_dict(),
             "init_args": self._init_args},
            filename
        )

    @staticmethod
    def load_dict(checkpoint, strict=False):
        init_args = checkpoint["init_args"]
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


@compile_mode("script")
class MLP(torch.nn.Module):

    def __init__(self,
                 irreps,
                 actv_type="gate",
                 actv_scalars={"e": "silu", "o": "tanh"},
                 actv_gates={"e": "relu", "o": "tanh"},
                 num_layers=3, resnet=True):

        super().__init__()

        assert len(irreps) == 2 or len(irreps) == num_layers+1

        self.irreps = [e3nn.o3.Irreps(irr).simplify() for irr in irreps]
        self.irreps_in = self.irreps[0]
        self.irreps_out = self.irreps[-1]

        args_dict = {'actv_type': actv_type,
                     'actv_scalars': actv_scalars,
                     'actv_gates': actv_gates}
        layer = PerceptronLayer

        if len(irreps) == 2:
            self.layers = torch.nn.ModuleList([layer(self.irreps_in, self.irreps_out, **args_dict)
                                               for _ in range(num_layers)])
        else:
            self.layers = torch.nn.ModuleList([layer(irr_in, irr_out, **args_dict)
                                               for irr_in, irr_out in zip(self.irreps, self.irreps[1:-1])])

        self.linear_out = e3nn.o3.Linear(irreps_in=self.layers[-1].irreps_out, irreps_out=self.irreps_out,
                                         internal_weights=True, shared_weights=True, biases=True)
        self.resnet = resnet

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            tmp = layer(x)
            if self.resnet and layer.irreps_in == layer.irreps_out:
                x = x + tmp
            else:
                x = tmp

        return self.linear_out(x)


@compile_mode("script")
class PerceptronLayer(torch.nn.Module):

    def __init__(self, irreps_in, irreps_out, **actv_kwargs):

        super().__init__()

        self.irreps_in = e3nn.o3.Irreps(irreps_in).simplify()
        self.irreps_out = e3nn.o3.Irreps(irreps_out).simplify()

        self.actv = Nonlinear(irreps_out=self.irreps_out, **actv_kwargs)
        self.linear = e3nn.o3.Linear(irreps_in=self.irreps_in, irreps_out=self.actv.irreps_required,
                                     internal_weights=True, shared_weights=True, biases=True)

    def forward(self, x):

        return self.actv(self.linear(x))


activations = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "relu": torch.relu,
}


@compile_mode("script")
class Nonlinear(torch.nn.Module):

    def __init__(self,
                 irreps_out,
                 actv_type="gate",
                 actv_scalars={"e": "silu", "o": "tanh"},
                 actv_gates={"e": "relu", "o": "tanh"}):

        if actv_type not in ("gate", "norm"):
            raise NotImplementedError("activation function type can only be gate or norm")

        super().__init__()

        self.irreps_out = e3nn.o3.Irreps(irreps_out)

        act = {
            1: activations[actv_scalars["e"]],
            -1: activations[actv_scalars["o"]],
        }
        act_gates = {
            1: activations[actv_gates["e"]],
            -1: activations[actv_gates["o"]],
        }

        irreps_scalars = e3nn.o3.Irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]
        )
        irreps_gated = e3nn.o3.Irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]
        )

        self.irreps_required = self.irreps_out

        if actv_type == "gate":
            ir = "0e"
            irreps_gates = e3nn.o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            self.actv = e3nn.nn.Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates=irreps_gates,
                act_gates=[act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated=irreps_gated,  # gated tensors
            )
            self.irreps_required = (irreps_scalars + irreps_gates + irreps_gated).simplify()
        else:
            self.actv = e3nn.nn.NormActivation(
                irreps_in=self.irreps_out,
                # norm is an even scalar
                scalar_nonlinearity=act[1],
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.actv(x)
