import os
import sys
import time
import torch
import argparse
import numpy as np
import ruamel_yaml as yaml
from collections import namedtuple
from pyscf import gto, lib
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepqc.scf.scf import DeepSCF, check_arg_list
from deepqc.train.model import QCNet
from deepqc.train.main import load_yaml


BOHR = 0.52917721092


Field = namedtuple("Field", ["name", "alias", "calc", "shape"])
SCF_FIELDS = [
    Field("e_hf", 
          ["ehf", "ene_hf", "e0"], 
          lambda mf: mf.energy_tot0(),
          "(nframe, 1)"),
    Field("e_cf", 
          ["ecf", "ene_cf", "e_tot", "etot", "ene", "energy", "e"],
          lambda mf: mf.e_tot,
          "(nframe, 1)"),
    Field("rdm",
          ["dm"],
          lambda mf: mf.make_rdm1(),
          "(nframe, nao, nao)"),
    Field("proj_dm",
          ["pdm"],
          lambda mf: mf.make_proj_rdms(flatten=True),
          "(nframe, natom, -1)"),
    Field("dm_eig",
          ["eig"],
          lambda mf: mf.make_eig(),
          "(nframe, natom, nproj)"),
    Field("conv", 
          ["converged", "convergence"], 
          lambda mf: mf.converged,
          "(nframe, 1)"),
    Field("mo_coef_occ", 
          ["mo_coeff_occ, orbital_coeff_occ"],
          lambda mf: mf.mo_coeff[:,mf.mo_occ>0].T,
          "(nframe, nao, -1)"),
    Field("mo_ene_occ", 
          ["mo_energy_occ, orbital_ene_occ"],
          lambda mf: mf.mo_energy[mf.mo_occ>0],
          "(nframe, -1)")
]
GRAD_FIELDS = [
    Field("f_hf", 
          ["fhf", "force_hf", "f0"], 
          lambda grad: - grad.get_hf() / BOHR,
          "(nframe, natom, 3)"),
    Field("f_cf", 
          ["fcf", "force_cf", "f_tot", "ftot", "force", "f"], 
          lambda grad: - grad.de / BOHR,
          "(nframe, natom, 3)"),
    Field("gdmx",
          ["grad_dm_x", "grad_pdm_x"],
          lambda grad: grad.make_grad_pdm_x(flatten=True) / BOHR,
          "(nframe,natom,3,natom,-1)"),
    Field("grad_vx",
          ["grad_eig_x", "geigx", "gvx"],
          lambda grad: grad.make_grad_eig_x() / BOHR,
          "(nframe,natom,3,natom,-1)"),
]
DEFAULT_FNAMES = ["e_cf", "e_hf", "dm_eig", "conv"]

DEFAULT_HF_ARGS = {
    "conv_tol": 1e-9
}

DEFAULT_SCF_ARGS = {
    "conv_tol": 1e-7,
    "level_shift": 0.1,
    "diis_space": 20
}

def load_xyz_files(file_list):
    if isinstance(file_list, str):
        file_list = [file_list]
    new_list = []
    for p in file_list:
        if os.path.splitext(p)[1] == '.xyz':
            new_list.append(p)
        else:
            with open(p) as f:
                new_list.extend(f.read().splitlines())
    return new_list


def parse_xyz(filename, basis='ccpvdz', verbose=0):
    with open(filename) as fp:
        natoms = int(fp.readline())
        comments = fp.readline()
        xyz_str = "".join(fp.readlines())
    mol = gto.Mole()
    mol.verbose = verbose
    mol.atom = xyz_str
    mol.basis  = basis
    mol.build(0,0,unit="Ang")
    return mol


def solve_mol(mol, model, fields,
              proj_basis=None, penalties=None, device=None,
              chkfile=None, verbose=0,
              **scf_args):
    if verbose:
        tic = time.time()

    cf = DeepSCF(mol, model, proj_basis, penalties, device)
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
        print(f"time of scf: {tac - tic}, converged: {cf.converged}")

    return meta, res


def parse_penalty(pnt_dict, basename="mol"):
    pnt_dict = pnt_dict.copy()
    pnt_type = pnt_dict.pop("type")
    if pnt_type.upper() == "DENSITY":
        from deepqc.scf.penalty import DensityPenalty
        suffix = pnt_dict.pop("suffix", "dm.npy").lstrip(".")
        basename = basename.rstrip(".xyz")
        dm_name = f"{basename}.{suffix}"
        return DensityPenalty(dm_name, **pnt_dict)
    if pnt_type.upper() == "COULOMB":
        from deepqc.scf.penalty import CoulombPenalty
        suffix = pnt_dict.pop("suffix", "dm.npy").lstrip(".")
        basename = basename.rstrip(".xyz")
        dm_name = f"{basename}.{suffix}"
        return CoulombPenalty(dm_name, **pnt_dict)
    else:
        raise KeyError(f"unknown penalty type: {pnt_type}")


def select_fields(names):
    names = [n.lower() for n in names]
    scfs  = [fd for fd in SCF_FIELDS 
                 if fd.name in names 
                 or any(al in names for al in fd.alias)]
    grads = [fd for fd in GRAD_FIELDS 
                 if fd.name in names 
                 or any(al in names for al in fd.alias)]
    return {"scf": scfs, "grad": grads}


def collect_fields(fields, meta, res_list):
    if isinstance(fields, dict):
        fields = fields["scf"] + fields["grad"]
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


def main(xyz_files, model_file="model.pth", basis='ccpvdz', 
         proj_basis=None, penalty_terms=None, device=None,
         dump_dir=None, dump_fields=DEFAULT_FNAMES, group=False, 
         scf_args=None, verbose=0):
    if model_file.upper() == "NONE":
        model = None
        default_scf_args = DEFAULT_HF_ARGS
    else:
        model = QCNet.load(model_file).double()
        default_scf_args = DEFAULT_SCF_ARGS
    # check arguments
    penalty_terms = check_arg_list(penalty_terms)
    if dump_dir is None:
        dump_dir = os.curdir
    if group:
        res_list = []
    if scf_args is None:
        scf_args = {}
    scf_args = {**default_scf_args, **scf_args}
    fields = select_fields(dump_fields)

    if verbose:
        print(f"starting calculation with OMP threads: {lib.num_threads()}")
        if verbose > 1:
            print(f"basis: {basis}")
            print(f"specified scf args:\n  {scf_args}")

    old_meta = None
    xyz_files = load_xyz_files(xyz_files)
    for fl in xyz_files:
        mol = parse_xyz(fl, basis=basis, verbose=verbose)
        penalties = [parse_penalty(pd, fl) for pd in penalty_terms]
        try:
            meta, result = solve_mol(mol, model, fields,
                                     proj_basis=proj_basis, penalties=penalties,
                                     device=device, verbose=verbose, **scf_args)
        except Exception as e:
            print(fl, 'failed! error:', e, file=sys.stderr)
            # continue
            raise
        if not group:
            sub_dir = os.path.join(dump_dir, os.path.splitext(os.path.basename(fl))[0])
            dump_meta(sub_dir, meta)
            dump_data(sub_dir, **collect_fields(fields, meta, [result]))
        else:
            if not res_list or np.all(meta == old_meta):
                res_list.append(result)
                old_meta = meta
            else:
                print(fl, 'meta does not match! saving previous results only.', file=sys.stderr)
                break
        if verbose:
            print(fl, 'finished')

    if group:
        dump_meta(dump_dir, meta)
        dump_data(dump_dir, **collect_fields(fields, meta, res_list))
        if verbose:
            print('group finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description="Calculate and save SCF energies and descriptors using given model.",
                argument_default=argparse.SUPPRESS)
    parser.add_argument("input", nargs="?",
                        help='the input yaml file for args')
    parser.add_argument("-x", "--xyz-files", nargs="*",
                        help="input xyz files")
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
    parser.add_argument("-G", "--group", action='store_true',
                        help="group results for all molecules, only works for same system")
    parser.add_argument("-v", "--verbose", type=int, choices=range(0,10),
                        help="output calculation information")
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
