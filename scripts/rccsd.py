#!/usr/bin/env python
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --mem=32G

import time
import numpy as np
from pyscf import gto, scf, cc

BOHR = 0.52917721092

def parse_xyz(filename, basis='ccpvdz', verbose=False, **kwargs):
    with open(filename) as fp:
        natoms = int(fp.readline())
        comments = fp.readline()
        xyz_str = "".join(fp.readlines())
    mol = gto.Mole()
    mol.verbose = 4 if verbose else 0
    mol.atom = xyz_str
    mol.basis = basis
    mol.set(**kwargs)
    mol.build(0,0,unit="Ang")
    return mol  

def calc_cc(mol):
    mf = scf.RHF(mol).run()
    if not mf.converged:
        return
    mycc = mf.CCSD().run()
    etot = mycc.e_tot
    ccdm = np.einsum('pi,ij,qj->pq', mf.mo_coeff, mycc.make_rdm1(), mf.mo_coeff.conj())
    grad = mycc.nuc_grad_method().kernel()
    return etot, ccdm, -grad/BOHR

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Calculate and save mp2 energy and mo_coeffs for given xyz files.")
    parser.add_argument("files", nargs="+", help="input xyz files")
    parser.add_argument("-d", "--dump-dir", help="dir of dumped files, default is same dir as xyz file")
    parser.add_argument("-v", "--verbose", action='store_true', help="output calculation information")
    parser.add_argument("-B", "--basis", default="ccpvdz", type=str, help="basis used to do the calculation")
    parser.add_argument("-C", "--charge", default=0, type=int, help="net charge of the molecule")
    args = parser.parse_args()
    
    if args.dump_dir is not None:
        os.makedirs(args.dump_dir, exist_ok = True)
    for fn in args.files:
        tic = time.time()
        mol = parse_xyz(fn, args.basis, args.verbose, charge=args.charge)
        res = calc_cc(mol)
        if res is None:
            print(fn, f"failed, SCF does not converge")
            continue
        etot, ccdm, force = res
        if args.dump_dir is None:
            dump_dir = os.path.dirname(fn)
        else:
            dump_dir = args.dump_dir
        dump = os.path.join(dump_dir, os.path.splitext(os.path.basename(fn))[0])
        np.save(dump+".energy.npy", [etot])
        np.save(dump+".dm.npy", ccdm)
        np.save(dump+".force.npy", force)
        print(fn, f"done, time = {time.time()-tic}")