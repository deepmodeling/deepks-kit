#!/usr/bin/env python
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 24:00:00
#SBATCH --mem=32G

import time
import numpy as np
from pyscf import gto, scf

BOHR = 0.52917721092


def parse_xyz(filename, basis='ccpvdz', **kwargs):
    with open(filename) as fp:
        natoms = int(fp.readline())
        comments = fp.readline()
        xyz_str = "".join(fp.readlines())
    mol = gto.Mole()
    mol.atom = xyz_str
    mol.basis = basis
    mol.set(**kwargs)
    mol.build(0,0,unit="Ang")
    return mol  


def calc_ccsd(mol, **scfargs):
    import pyscf.cc
    mf = scf.RHF(mol).run(**scfargs)
    if not mf.converged:
        return
    mycc = mf.CCSD().run()
    etot = mycc.e_tot
    grad = mycc.nuc_grad_method().kernel()
    ccdm = np.einsum('pi,ij,qj->pq', mf.mo_coeff, mycc.make_rdm1(), mf.mo_coeff.conj())
    return etot, -grad/BOHR, ccdm


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Calculate and save mp2 energy and mo_coeffs for given xyz files.")
    parser.add_argument("files", nargs="+", help="input xyz files")
    parser.add_argument("-d", "--dump-dir", help="dir of dumped files, default is same dir as xyz file")
    parser.add_argument("-v", "--verbose", default=1, type=int, help="output calculation information")
    parser.add_argument("-B", "--basis", default="ccpvdz", type=str, help="basis used to do the calculation")
    parser.add_argument("-C", "--charge", default=0, type=int, help="net charge of the molecule")
    parser.add_argument("--scf-input", help="yaml file to specify scf arguments")
    args = parser.parse_args()
    
    scfargs = {}
    if args.scf_input is not None:
        import ruamel.yaml as yaml
        with open(args.scf_input, 'r') as fp:
            scfargs = yaml.safe_load(fp)        
    if args.dump_dir is not None:
        os.makedirs(args.dump_dir, exist_ok = True)

    for fn in args.files:
        tic = time.time()
        mol = parse_xyz(fn, args.basis, verbose=args.verbose, charge=args.charge)
        res = calc_ccsd(mol, **scfargs)
        if res is None:
            print(fn, f"failed, SCF does not converge")
            continue
        etot, force, rdm = res
        if args.dump_dir is None:
            dump_dir = os.path.dirname(fn)
        else:
            dump_dir = args.dump_dir
        dump = os.path.join(dump_dir, os.path.splitext(os.path.basename(fn))[0])
        np.save(dump+".energy.npy", [etot])
        np.save(dump+".force.npy", force)
        if rdm is not None:
            np.save(dump+".dm.npy", rdm)
        if args.verbose:
            print(fn, f"done, time = {time.time()-tic}")