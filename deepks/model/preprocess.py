
import sys
import numpy as np

import e3nn.o3

from deepks.model.model import CorrNet
from deepks.model.model_enn import CorrNet as CorrNetEquiv
from deepks.utils import load_elem_table


def compute_data_stat(g_reader, symm_sections=None):

    all_dm = np.concatenate([r.data_dm.reshape(-1, r.ndesc) for r in g_reader.readers])
    if symm_sections is None:
        all_mean, all_std = all_dm.mean(0), all_dm.std(0)
    else:
        assert sum(symm_sections) == all_dm.shape[-1]
        dm_shells = np.split(all_dm, np.cumsum(symm_sections)[:-1], axis=-1)
        mean_shells = [d.mean().repeat(s) for d, s in zip(dm_shells, symm_sections)]
        std_shells = [d.std().repeat(s) for d, s in zip(dm_shells, symm_sections)]
        all_mean = np.concatenate(mean_shells, axis=-1)
        all_std = np.concatenate(std_shells, axis=-1)
    return all_mean, all_std


def compute_prefitting(g_reader, shift=None, scale=None, ridge_alpha=1e-8, symm_sections=None):
    if shift is None or scale is None:
        all_mean, all_std = compute_data_stat(g_reader, symm_sections=symm_sections)
        if shift is None:
            shift = all_mean
        if scale is None:
            scale = all_std
    all_sdm = np.concatenate([((r.data_dm - shift) / scale).sum(1) for r in g_reader.readers])
    all_natm = np.concatenate([[float(r.data_dm.shape[1])]*r.data_dm.shape[0] for r in g_reader.readers])
    if symm_sections is not None: # in this case ridge alpha cannot be 0
        assert sum(symm_sections) == all_sdm.shape[-1]
        sdm_shells = np.split(all_sdm, np.cumsum(symm_sections)[:-1], axis=-1)
        all_sdm = np.stack([d.sum(-1) for d in sdm_shells], axis=-1)
    # build feature matrix
    X = np.concatenate([all_sdm, all_natm.reshape(-1,1)], -1)
    y = np.concatenate([r.data_ec for r in g_reader.readers])
    I = np.identity(X.shape[1])
    I[-1,-1] = 0 # do not punish the bias term
    # solve ridge reg
    coef = np.linalg.solve(X.T @ X + ridge_alpha * I, X.T @ y).reshape(-1)
    weight, bias = coef[:-1], coef[-1]
    if symm_sections is not None:
        weight = np.concatenate([w.repeat(s) for w, s in zip(weight, symm_sections)], axis=-1)
    return weight, bias


def preprocess(model, g_reader,
               preshift=True, prescale=False, prescale_sqrt=False, prescale_clip=0,
               prefit=True, prefit_ridge=10, prefit_trainable=False):
    shift = model.input_shift.cpu().detach().numpy()
    scale = model.input_scale.cpu().detach().numpy()
    symm_sec = model.shell_sec # will be None if no embedding
    prefit_trainable = prefit_trainable and symm_sec is None # no embedding
    if preshift or prescale:
        davg, dstd = compute_data_stat(g_reader, symm_sec)
        if preshift:
            shift = davg
        if prescale:
            scale = dstd
            if prescale_sqrt:
                scale = np.sqrt(scale)
            if prescale_clip:
                scale = scale.clip(prescale_clip)
        model.set_normalization(shift, scale)
    if prefit:
        weight, bias = compute_prefitting(g_reader,
                                          shift=shift, scale=scale,
                                          ridge_alpha=prefit_ridge, symm_sections=symm_sec)
        model.set_prefitting(weight, bias, trainable=prefit_trainable)


# -- enn related --

def compute_data_stat_enn(g_reader, irreps_str):

    irreps = e3nn.o3.Irreps(irreps_str)
    Ndesc = irreps.count(e3nn.o3.Irrep('0e'))

    all_dm = np.concatenate([r.data_dm[..., :Ndesc].reshape(-1, Ndesc) for r in g_reader.readers])
    mean = np.mean(all_dm, dim=0)

    return mean


def compute_prefitting_enn(g_reader, irreps_str=None, ridge_alpha=1e-8, preshift=False):

    irreps = e3nn.o3.Irreps(irreps_str)
    ndesc = irreps.count(e3nn.o3.Irrep('0e'))

    shift = compute_data_stat_enn(g_reader, irreps_str) if preshift else 0.

    all_sdm = np.concatenate([(r.data_dm[..., :ndesc] - shift).sum(-2) for r in g_reader.readers])
    all_natm = np.concatenate([[float(r.data_dm.shape[1])]*r.data_dm.shape[0] for r in g_reader.readers])

    # build feature matrix
    X = np.concatenate([all_sdm, all_natm.reshape(-1, 1)], -1)
    y = np.concatenate([r.data_ec for r in g_reader.readers])
    I = np.identity(X.shape[1])
    I[-1, -1] = 0  # do not punish the bias term
    # solve ridge reg
    coef = np.linalg.solve(X.T @ X + ridge_alpha * I, X.T @ y).reshape(-1)
    weight, bias = coef[:-1], coef[-1]

    return weight, bias


def set_desc_lmax(g_reader, lmax=-1):

    if lmax < 0: return

    import e3nn.o3
    irreps_str = g_reader.readers[0].irreps_str  # for now assume all configurations share the same irreps
    irreps = e3nn.o3.Irreps([(mul, ir) for mul, ir in e3nn.o3.Irreps(irreps_str) if ir.l <= lmax])
    ndesc = irreps.dim
    irreps_str = str(irreps)

    # cut descriptor and reset irreps
    for r in g_reader.readers:
        r.data_dm = r.data_dm[..., :ndesc]
        r.t_data["desc"] = r.t_data["desc"][..., :ndesc]
        r.irreps_str = irreps_str
        r.ndesc = ndesc
    g_reader.ndesc = ndesc


def preprocess_enn(model, g_reader, ridge_alpha=10, prefit=True, prefit_trainable=False, preshift=True):

    if not (prefit or preshift): return

    irreps = g_reader.readers[0].irreps_str  # for now assume all configurations share the same irreps
    if prefit:
        weight, bias = compute_prefitting_enn(g_reader, irreps_str=irreps, ridge_alpha=ridge_alpha, preshift=preshift)
        model.set_prefitting(weight, bias, trainable=prefit_trainable)
    if preshift:
        mean = compute_data_stat_enn(g_reader, irreps)
        model.set_preshift(mean)


# -- model related --

def fit_elem_const(g_reader, test_reader=None, elem_table=None, ridge_alpha=0.):
    if elem_table is None:
        elem_table = g_reader.compute_elem_const(ridge_alpha)
    elem_list, elem_const = elem_table
    g_reader.collect_elems(elem_list)
    g_reader.subtract_elem_const(elem_const)
    if test_reader is not None:
        test_reader.collect_elems(elem_list)
        test_reader.subtract_elem_const(elem_const)
    return elem_table


def make_model(g_reader, test_reader, restart, model_args, preprocess_args, fit_elem=False):

    model_type = model_args["model_type"]
    del model_args["model_type"]

    if model_type == "invariant":
        prep = preprocess
        if restart is not None:
            model = CorrNet.load(restart)
            if model.elem_table is not None:
                fit_elem_const(g_reader, test_reader, model.elem_table)
        else:
            input_dim = g_reader.ndesc
            if model_args.get("input_dim", input_dim) != input_dim:
                print(f"# `input_dim` in `model_args` does not match data",
                      f"({input_dim}).", "Use the one in data.", file=sys.stderr)
            model_args["input_dim"] = input_dim
            if fit_elem:
                elem_table = model_args.get("elem_table", None)
                if isinstance(elem_table, str):
                    elem_table = load_elem_table(elem_table)
                elem_table = fit_elem_const(g_reader, test_reader, elem_table)
                model_args["elem_table"] = elem_table
            model = CorrNet(**model_args).double()
    elif model_type == "equivariant":
        prep = preprocess_enn
        if restart is not None:
            model = CorrNetEquiv.load(restart)
        else:
            irreps_str = g_reader.readers[0].irreps_str
            model_args["irreps_in"] = irreps_str
            if "irreps_hidden" not in model_args.keys():
                irreps_hidden = None
                if "hidden_features" in model_args.keys():
                    irreps_hidden = str(e3nn.o3.Irreps([(model_args["hidden_features"],
                                                         ir) for _, ir in e3nn.o3.Irreps(irreps_str)]))
                model_args["irreps_hidden"] = irreps_hidden
            model = CorrNetEquiv(**model_args).double()
    else:
        raise RuntimeError("model_type can only be invariant or equivariant")

    prep(model, g_reader, **preprocess_args)

    return model
