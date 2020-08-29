import os
import sys
import time
import torch
import argparse
import numpy as np
import ruamel_yaml as yaml
from pyscf import gto, lib
try:
    import deepqc
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepqc.scf.scf import DeepSCF
from deepqc.scf.fields import select_fields
from deepqc.scf.penalty import select_penalty
from deepqc.train.model import QCNet
from deepqc.utils import check_list, flat_file_list
from deepqc.utils import is_xyz, load_sys_paths
from deepqc.utils import load_yaml, load_array
from deepqc.utils import get_with_prefix

DEFAULT_FNAMES = ["e_cf", "e_hf", "dm_eig", "conv"]

DEFAULT_HF_ARGS = {
    "conv_tol": 1e-9
}

DEFAULT_SCF_ARGS = {
    "conv_tol": 1e-7,
    "level_shift": 0.1,
    "diis_space": 20
}


def solve_mol(mol, model, fields,
              proj_basis=None, penalties=None, device=None,
              chkfile=None, verbose=0,
              **scf_args):
    if verbose:
        tic = time.time()

    cf = DeepSCF(mol, model, 
                 proj_basis=proj_basis, 
                 penalties=penalties, 
                 device=device)
    cf.set(chkfile=chkfile)
    cf.set(**scf_args)
    cf.kernel()

    natom = mol.natm
    nao = mol.nao
    nproj = sum(cf.shell_sec)
    meta = np.array([natom, nao, nproj])

    res = {}
    for fd in fields["scf"]:
        res[fd.name] = fd.calc(cf)
    if fields["grad"]:
        gd = cf.nuc_grad_method().run()
    for fd in fields["grad"]:
        res[fd.name] = fd.calc(gd)
    
    if verbose:
        tac = time.time()
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
    base = path.rstrip(".xyz")
    label_paths = {lb: get_with_prefix(lb, base, prefer=".npy") for lb in labels}
    # if xyz, will yield single frame. Assume all labels are single frame
    if is_xyz(path):
        atom = path
        label_dict = {lb: load_array(label_paths[lb]) for lb in labels}
        yield atom, label_dict
        return
    # a folder contains multiple frames data, yield one by one
    else:
        assert os.path.isdir(path), f"system {path} is neither .xyz or dir"
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
            elements = np.loadtxt(os.path.join(path, "type.raw"), dtype='str')\
                         .reshape(1,-1).repeat(nframes, axis=0)
        for i in range(nframes):
            atom = [[e,c] for e,c in zip(elements[i], coords[i])]
            label_dict = {lb: all_labels[lb][i] for lb in labels}
            yield atom, label_dict


def build_mol(atom, basis='ccpvdz', verbose=0, **kwargs):
    # build a molecule using given atom input
    # set the default basis to cc-pVDZ and use input unit 'Ang"
    mol = gto.Mole()
    # change minimum max memory to 16G
    # mol.max_memory = max(16000, mol.max_memory) 
    mol.set(**kwargs)
    mol.verbose = verbose
    mol.atom = atom
    mol.basis = basis
    mol.build(0,0,unit="Ang")
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
        model = QCNet.load(model_file).double()
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

    old_meta = None
    res_list = []
    systems = load_sys_paths(systems)

    for fl in systems:
        fl = fl.rstrip(os.path.sep)
        for atom, labels in system_iter(fl, label_names):
            mol = build_mol(atom, basis=basis, verbose=verbose, **mol_args)
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
            sub_dir = os.path.join(dump_dir, os.path.basename(fl).rstrip(".xyz"))
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


def cli():
    parser = argparse.ArgumentParser(
                description="Calculate and save SCF energies and descriptors using given model.",
                argument_default=argparse.SUPPRESS)
    parser.add_argument("input", nargs="?",
                        help='the input yaml file for args')
    parser.add_argument("-s", "--systems", nargs="*",
                        help="input molecule systems, can be xyz files or folders with npy data")
    parser.add_argument("-m", "--model-file",
                        help="file of the trained model")
    parser.add_argument("-B", "--basis",
                        help="basis set used to solve the model") 
    parser.add_argument("-P", "--proj_basis",
                        help="basis set used to project dm, must match with model")   
    parser.add_argument("-D", "--device",
                        help="device name used in nn model inference")               
    parser.add_argument("-d", "--dump-dir",
                        help="dir of dumped files")
    parser.add_argument("-F", "--dump-fields", nargs="*",
                        help="fields to be dumped into the folder") 
    group0 = parser.add_mutually_exclusive_group()   
    group0.add_argument("-G", "--group", action='store_true', dest="group",
                        help="group results for all systems, only works for same system")
    group0.add_argument("-NG", "--no-group", action='store_false', dest="group",
                        help="Do not group results for different systems (default behavior)")
    parser.add_argument("-v", "--verbose", type=int, choices=range(0,10),
                        help="output calculation information")
    parser.add_argument("-X", "--scf-xc",
                        help="base xc functional used in scf equation, default is HF")        
    parser.add_argument("--scf-conv-tol", type=float,
                        help="converge threshold of scf iteration")
    parser.add_argument("--scf-conv-tol-grad", type=float,
                        help="gradient converge threshold of scf iteration")
    parser.add_argument("--scf-max-cycle", type=int,
                        help="max number of scf iteration cycles")
    parser.add_argument("--scf-diis-space", type=int,
                        help="subspace dimension used in diis mixing")
    parser.add_argument("--scf-level-shift", type=float,
                        help="level shift used in scf calculation")

    args = parser.parse_args()

    scf_args={}
    for k, v in vars(args).copy().items():
        if k.startswith("scf_"):
            scf_args[k[4:]] = v
            delattr(args, k)

    if hasattr(args, "input"):
        argdict = load_yaml(args.input)
        del args.input
        argdict.update(vars(args))
        argdict["scf_args"].update(scf_args)
    else:
        argdict = vars(args)
        argdict["scf_args"] = scf_args

    main(**argdict)


if __name__ == "__main__":
    cli()