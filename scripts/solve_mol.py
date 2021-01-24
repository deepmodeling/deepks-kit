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
    mol.spin = mol.nelectron % 2
    mol.build(0,0,unit="Ang")
    return mol  


def get_method(name: str):
    lname = name.lower()
    if lname == "hf":
        return calc_hf
    if lname[:3] == "dft":
        xc = lname.split("@")[1]
        return lambda mol, **scfargs: calc_dft(mol, xc, **scfargs)
    if lname == "mp2":
        return calc_mp2
    if lname == "ccsd":
        return calc_ccsd
    if lname in ("ccsd_t", "ccsd-t", "ccsd(t)"):
        return calc_ccsd_t
    raise ValueError(f"Unknown calculation method: {name}")

def calc_hf(mol, **scfargs):
    mf = scf.HF(mol).run(**scfargs)
    if not mf.converged:
        raise RuntimeError("SCF not converged!")
    etot = mf.e_tot
    grad = mf.nuc_grad_method().kernel()
    rdm = mf.make_rdm1()
    return etot, -grad/BOHR, rdm

def calc_dft(mol, xc="pbe", **scfargs):
    from pyscf import dft
    mf = dft.KS(mol, xc).run(**scfargs)
    if not mf.converged:
        raise RuntimeError("SCF not converged!")
    etot = mf.e_tot
    grad = mf.nuc_grad_method().kernel()
    rdm = mf.make_rdm1()
    return etot, -grad/BOHR, rdm

def calc_mp2(mol, **scfargs):
    import pyscf.mp
    mf = scf.HF(mol).run(**scfargs)
    if not mf.converged:
        raise RuntimeError("SCF not converged!")
    postmf = pyscf.mp.MP2(mf).run()
    etot = postmf.e_tot
    grad = postmf.nuc_grad_method().kernel()
    return etot, -grad/BOHR, None

def calc_ccsd(mol, **scfargs):
    import pyscf.cc
    mf = scf.HF(mol).run(**scfargs)
    if not mf.converged:
        raise RuntimeError("SCF not converged!")
    mycc = mf.CCSD().run()
    etot = mycc.e_tot
    grad = mycc.nuc_grad_method().kernel()
    ccdm = np.einsum('...pi,...ij,...qj->...pq', mf.mo_coeff, mycc.make_rdm1(), mf.mo_coeff.conj())
    return etot, -grad/BOHR, ccdm

def calc_ccsd_t(mol, **scfargs):
    import pyscf.cc
    import pyscf.grad.ccsd_t as ccsd_t_grad
    mf = scf.HF(mol).run(**scfargs)
    if not mf.converged:
        raise RuntimeError("SCF not converged!")
    mycc = mf.CCSD().run()
    et_correction = mycc.ccsd_t()
    etot = mycc.e_tot + et_correction
    grad = ccsd_t_grad.Gradients(mycc).kernel()
    return etot, -grad/BOHR, None


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Calculate and save mp2 energy and mo_coeffs for given xyz files.")
    parser.add_argument("files", nargs="+", help="input xyz files")
    parser.add_argument("-d", "--dump-dir", help="dir of dumped files, default is same dir as xyz file")
    parser.add_argument("-v", "--verbose", default=1, type=int, help="output calculation information")
    parser.add_argument("-B", "--basis", default="ccpvdz", type=str, help="basis used to do the calculation")
    parser.add_argument("-C", "--charge", default=0, type=int, help="net charge of the molecule")
    parser.add_argument("-M", "--method", default="ccsd", help="method used to do the calculation. support MP2, CCSD and CCSD(T)")
    parser.add_argument("--scf-input", help="yaml file to specify scf arguments")
    args = parser.parse_args()
    
    scfargs = {}
    if args.scf_input is not None:
        import ruamel.yaml as yaml
        with open(args.scf_input, 'r') as fp:
            scfargs = yaml.safe_load(fp)        
    if args.dump_dir is not None:
        os.makedirs(args.dump_dir, exist_ok = True)
    calculator = get_method(args.method)

    for fn in args.files:
        tic = time.time()
        mol = parse_xyz(fn, args.basis, verbose=args.verbose, charge=args.charge)
        try:
            res = calculator(mol, **scfargs)
        except RuntimeError as err:
            print(fn, f"failed, {err}")
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