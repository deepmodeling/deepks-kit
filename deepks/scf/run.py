import os
import sys
import time
import numpy as np
import torch
from pyscf import gto, lib
try:
    import deepks
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepks.scf.scf import DSCF, UDSCF
from deepks.scf.fields import select_fields
from deepks.scf.penalty import select_penalty
from deepks.model.model import CorrNet
from deepks.utils import check_list, flat_file_list
from deepks.utils import is_xyz, load_sys_paths
from deepks.utils import load_yaml, load_array
from deepks.utils import get_sys_name, get_with_prefix

DEFAULT_FNAMES = {"e_tot", "e_base", "dm_eig", "conv"}

DEFAULT_HF_ARGS = {
    "conv_tol": 1e-9
}

DEFAULT_SCF_ARGS = {
    "conv_tol": 1e-7,
    "level_shift": 0.1,
    "diis_space": 20
}

MOL_ATTRIBUTE = {"charge"} # basis, symmetry, and more

def solve_mol(mol, model, fields,
              proj_basis=None, penalties=None, device=None,
              chkfile=None, verbose=0,
              **scf_args):
    
    tic = time.time()

    SCFcls = DSCF if mol.spin == 0 else UDSCF
    cf = SCFcls(mol, model, 
                proj_basis=proj_basis, 
                penalties=penalties, 
                device=device)
    cf.set(chkfile=chkfile)
    cf.set(**scf_args)
    cf.kernel()

    natom = mol.natm
    nao = mol.nao
    nproj = cf.nproj
    meta = np.array([natom, nao, nproj])

    res = {}
    for fd in fields["scf"]:
        res[fd.name] = fd.calc(cf)
    if fields["grad"]:
        gd = cf.nuc_grad_method().run()
        for fd in fields["grad"]:
            res[fd.name] = fd.calc(gd)
    
    tac = time.time()
    if verbose:
        print(f"time of scf: {tac - tic:6.2f}s, converged:   {cf.converged}")

    return meta, res


def get_required_labels(fields=None, penalty_dicts=None):
    field_labels   = [check_list(f.required_labels)
                        for f in check_list(fields)]
    penalty_labels = [check_list(p.get("required_labels", 
                                       select_penalty(p["type"]).required_labels))
                        for p in check_list(penalty_dicts)]
    return set(sum(field_labels + penalty_labels, []))


def system_iter(path, labels=None):
    """
    return an iterator that gives atoms and required labels each time
    path: either an xyz file, or a folder contains (atom.npy | (coord.npy @ type.raw))
    labels: a set contains required label names, will be load by $base[.|/]$label.npy
    $base will be the basename of the xyz file (followed by .) or the folder (followed by /)
    """
    if labels is None:
        labels = set()
    base = get_sys_name(path)
    attr_paths = {at: get_with_prefix(at, base, ".npy", True) for at in MOL_ATTRIBUTE}
    attr_paths = {k: v for k, v in attr_paths.items() if v is not None}
    attrs = attr_paths.keys()
    label_paths = {lb: get_with_prefix(lb, base, prefer=".npy") for lb in labels}
    # if xyz, will yield single frame. Assume all labels are single frame
    if is_xyz(path):
        atom = path
        attr_dict = {at: load_array(attr_paths[at]) for at in attrs}
        label_dict = {lb: load_array(label_paths[lb]) for lb in labels}
        yield atom, attr_dict, label_dict
        return
    # a folder contains multiple frames data, yield one by one
    else:
        assert os.path.isdir(path), f"system {path} is neither .xyz or dir"
        all_attrs = {at: load_array(attr_paths[at]) for at in attrs}
        all_labels = {lb: load_array(label_paths[lb]) for lb in labels}
        try:
            atom_array = load_array(get_with_prefix("atom", path, prefer=".npy"))
            assert len(atom_array.shape) == 3 and atom_array.shape[2] == 4, atom_array.shape
            nframes = atom_array.shape[0]
            elements = np.rint(atom_array[:, :, 0]).astype(int)
            coords = atom_array[:, :, 1:]
        except FileNotFoundError:
            coords = load_array(get_with_prefix("coord", path, prefer=".npy"))
            assert len(coords.shape) == 3 and coords.shape[2] == 3, coords.shape
            nframes = coords.shape[0]
            elements = np.loadtxt(os.path.join(path, "type.raw"), dtype=str)\
                         .reshape(1,-1).repeat(nframes, axis=0)
        for i in range(nframes):
            atom = [[e,c] for e,c in zip(elements[i], coords[i])]
            attr_dict = {at: (all_attrs[at][i] 
                                if all_attrs[at].ndim > 0
                                and all_attrs[at].shape[0] == nframes
                                else all_attrs[at]) 
                         for at in attrs}
            label_dict = {lb: all_labels[lb][i] for lb in labels}
            yield atom, attr_dict, label_dict
        return


def build_mol(atom, basis='ccpvdz', verbose=0, **kwargs):
    # build a molecule using given atom input
    # set the default basis to cc-pVDZ and use input unit 'Ang"
    mol = gto.Mole()
    # change minimum max memory to 16G
    # mol.max_memory = max(16000, mol.max_memory) 
    mol.unit = "Ang"
    mol.atom = atom
    mol.basis = basis
    mol.verbose = verbose
    mol.__dict__.update(kwargs)
    mol.spin = mol.nelectron % 2
    mol.build(0,0)
    return mol


def build_penalty(pnt_dict, label_dict={}):
    pnt_dict = pnt_dict.copy()
    pnt_type = pnt_dict.pop("type")
    PenaltyClass = select_penalty(pnt_type)
    label_names = pnt_dict.pop("required_labels", PenaltyClass.required_labels)
    label_arrays = [label_dict[lb] for lb in check_list(label_names)]
    return PenaltyClass(*label_arrays, **pnt_dict)


def make_labels(res, lbl, label_fields):
    if isinstance(label_fields, dict):
        label_fields = label_fields["label"]
    for fd in label_fields:
        res[fd.name] = fd.calc(res, lbl)
    return res


def collect_fields(fields, meta, res_list):
    if isinstance(fields, dict):
        fields = sum(fields.values(), [])
    if isinstance(res_list, dict):
        res_list = [res_list]
    nframe = len(res_list)
    natom, nao, nproj = meta
    res_dict = {}
    for fd in fields:
        fd_res = np.array([res[fd.name] for res in res_list])
        if fd.shape:
            fd_shape = eval(fd.shape, {}, locals())
            fd_res = fd_res.reshape(fd_shape)
        res_dict[fd.name] = fd_res
    return res_dict


def dump_meta(dir_name, meta):
    os.makedirs(dir_name, exist_ok = True)
    np.savetxt(os.path.join(dir_name, 'system.raw'), 
               np.reshape(meta, (1,-1)), 
               fmt = '%d', header = 'natom nao nproj')


def dump_data(dir_name, **data_dict):
    os.makedirs(dir_name, exist_ok = True)
    for name, value in data_dict.items():
        np.save(os.path.join(dir_name, f'{name}.npy'), value)


def main(systems, model_file="model.pth", basis='ccpvdz', 
         proj_basis=None, penalty_terms=None, device=None,
         dump_dir=".", dump_fields=DEFAULT_FNAMES, group=False, 
         mol_args=None, scf_args=None, verbose=0):
    if model_file is None or model_file.upper() == "NONE":
        model = None
        default_scf_args = DEFAULT_HF_ARGS
    else:
        model = CorrNet.load(model_file).double()
        default_scf_args = DEFAULT_SCF_ARGS

    # check arguments
    penalty_terms = check_list(penalty_terms)
    if mol_args is None: mol_args = {}
    if scf_args is None: scf_args = {}
    scf_args = {**default_scf_args, **scf_args}
    fields = select_fields(dump_fields)
    # check label names from label fields and penalties
    label_names = get_required_labels(fields["label"], penalty_terms)

    if verbose:
        print(f"starting calculation with OMP threads: {lib.num_threads()}",
              f"and max memory: {lib.param.MAX_MEMORY}")
        if verbose > 1:
            print(f"basis: {basis}")
            print(f"specified scf args:\n  {scf_args}")

    meta = old_meta = None
    res_list = []
    systems = load_sys_paths(systems)

    for fl in systems:
        fl = fl.rstrip(os.path.sep)
        for atom, attrs, labels in system_iter(fl, label_names):
            mol_input = {**mol_args, "verbose":verbose, 
                        "atom": atom, "basis": basis,  **attrs}
            mol = build_mol(**mol_input)
            penalties = [build_penalty(pd, labels) for pd in penalty_terms]
            try:
                meta, result = solve_mol(mol, model, fields,
                                         proj_basis=proj_basis, penalties=penalties,
                                         device=device, verbose=verbose, **scf_args)
                result = make_labels(result, labels, fields["label"])
            except Exception as e:
                print(fl, 'failed! error:', e, file=sys.stderr)
                # continue
                raise
            if group and old_meta is not None and np.any(meta != old_meta):
                break
            res_list.append(result)

        if not group:
            sub_dir = os.path.join(dump_dir, get_sys_name(os.path.basename(fl)))
            dump_meta(sub_dir, meta)
            dump_data(sub_dir, **collect_fields(fields, meta, res_list))
            res_list = []
        elif old_meta is not None and np.any(meta != old_meta):
            print(fl, 'meta does not match! saving previous results only.', file=sys.stderr)
            break
        old_meta = meta
        if verbose:
            print(fl, 'finished')

    if group:
        dump_meta(dump_dir, meta)
        dump_data(dump_dir, **collect_fields(fields, meta, res_list))
        if verbose:
            print('group finished')


if __name__ == "__main__":
    from deepks.main import scf_cli as cli
    cli()