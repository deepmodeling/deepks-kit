import os
import sys
import time
import torch
import argparse
import numpy as np
from pyscf import gto
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepqc.scf.scf import DeepSCF
from deepqc.train.model import QCNet


def load_xyz_files(file_list):
    new_list = []
    for p in file_list:
        if os.path.splitext(p)[1] == '.xyz':
            new_list.append(p)
        else:
            with open(p) as f:
                new_list.extend(f.read().splitlines())
    return new_list


def parse_xyz(filename, basis='ccpvtz', verbose=0):
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


def solve_mol(mol, model, 
              conv_tol=1e-9, conv_tol_grad=None,
              chkfile=None, verbose=0):
    if verbose:
        tic = time.time()
    cf = DeepSCF(mol, model)
    cf.conv_tol = conv_tol
    cf.conv_tol_grad = conv_tol_grad
    if chkfile:
        cf.set(chkfile=chkfile)
    ecf = cf.scf()
    if verbose:
        tac = time.time()
        print(f"time of scf: {tac - tic}, converged: {cf.converged}")
    natom = mol.natm
    nao = mol.nao
    nproj = sum(cf.shell_sec)
    meta = np.array([natom, nao, nproj])
    dm = cf.make_rdm1()
    eig = cf.make_eig(dm)
    ehf = cf.energy_tot0(dm)
    return meta, ehf, ecf, dm, eig, cf.converged


def check_fields(fields, results):
    nframe = len(results) if isinstance(results, list) else 1
    meta, ehf, ecf, dm, eig, conv = zip(*results)
    natom, nao, nproj = meta[0]
    res_dict = {}
    if 'e_hf' in fields or 'ehf' in fields:
        res_dict['e_hf'] = np.reshape(ehf, (nframe,1))
    if 'e_cf' in fields or 'ecf' in fields:
        res_dict['e_cf'] = np.reshape(ecf, (nframe,1))
    if 'dm' in fields or 'rdm' in fields:
        res_dict['rdm'] = np.reshape(dm, (nframe,nao,nao))
    if 'eig' in fields or 'dm_eig' in fields:
        res_dict['dm_eig'] = np.reshape(eig, (nframe,natom,nproj))
    if 'conv' in fields or 'convergence' in fields:
        res_dict['conv'] = np.reshape(conv, (nframe,1))
    return res_dict


def dump_data(dir_name, meta, **data_dict):
    os.makedirs(dir_name, exist_ok = True)
    np.savetxt(os.path.join(dir_name, 'system.raw'), 
               np.reshape(meta, (1,-1)), 
               fmt = '%d', header = 'natom nao nproj')
    for name, value in data_dict.items():
        np.save(os.path.join(dir_name, f'{name}.npy'), value)


def main(xyz_files, model_file, 
         dump_dir=None, dump_fields=['e_cf'], group=False, 
         conv_tol=1e-9, conv_tol_grad=None,
         verbose=0):

    model = QCNet.load(model_file).double()
    if dump_dir is None:
        dump_dir = os.curdir
    if group:
        results = []
        
    xyz_files = load_xyz_files(xyz_files)
    for fl in xyz_files:
        mol = parse_xyz(fl, verbose=verbose)
        try:
            result = solve_mol(mol, model, 
                               conv_tol=conv_tol, conv_tol_grad=conv_tol_grad,
                               verbose=verbose)
        except Exception as e:
            print(fl, 'failed! error:', e, file=sys.stderr)
            continue
        if not group:
            meta = result[0]
            sub_dir = os.path.join(dump_dir, os.path.splitext(os.path.basename(fl))[0])
            dump_data(sub_dir, meta, **check_fields(dump_fields, [result]))
        else:
            if not results or all(result[0] == results[0][0]):
                results.append(result)
            else:
                print(fl, 'meta does not match! saving previous results only.', file=sys.stderr)
                break
        if verbose:
            print(fl, 'finished')

    if group:
        meta = results[0][0]
        dump_data(dump_dir, meta, **check_fields(dump_fields, results))
        if verbose:
            print('group finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and save SCF energies and descriptors using given model.")
    parser.add_argument("xyz_files", nargs="+", 
                        help="input xyz files")
    parser.add_argument("-m", "--model-file", default='model.pth', 
                        help="file of the trained model")
    parser.add_argument("-d", "--dump-dir", default='.', 
                        help="dir of dumped files")
    parser.add_argument("-F", "--dump-fields", nargs="+", default=['e_hf', 'e_cf', 'dm_eig', 'conv'],
                        help="fields to be dumped into the folder")    
    parser.add_argument("-G", "--group", action='store_true',
                        help="group results for all molecules, only works for same system")
    parser.add_argument("-v", "--verbose", default=0, type=int, choices=range(0,11),
                        help="output calculation information")
    parser.add_argument("--conv-tol", default=1e-9, type=float,
                        help="converge threshold of scf iteration")
    parser.add_argument("--conv-tol-grad", default=None, type=float,
                        help="gradient converge threshold of scf iteration")
    args = parser.parse_args()
    
    main(**vars(args))