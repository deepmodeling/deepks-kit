#!/usr/bin/env python
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 24:00:00
#SBATCH --mem=32G

import time
import numpy as np
from pyscf import gto, scf

BOHR = 0.52917721092

_NO_FORCE = False
_NO_DM = False
_MUST_UNRES = False
_USE_NEWTON = False

def parse_xyz(filename, basis='ccpvdz', **kwargs):
    with open(filename) as fp:
        natoms = int(fp.readline())
        comments = fp.readline()
        xyz_str = "".join(fp.readlines())
    mol = gto.Mole()
    mol.atom = xyz_str
    mol.basis = basis
    mol.set(**kwargs)
    if "spin" not in kwargs:
        mol.spin = mol.nelectron % 2
    mol.build(0,0,unit="Ang")
    return mol  


def get_method(name: str):
    lname = name.lower()
    if lname == "hf":
        return calc_hf
    if lname[:3] == "dft":
        xc = lname.split("@")[1] if "@" in lname else "pbe"
        return lambda mol, **scfargs: calc_dft(mol, xc, **scfargs)
    if lname == "mp2":
        return calc_mp2
    if lname == "ccsd":
        return calc_ccsd
    if lname.startswith(("ccsd_t", "ccsd-t", "ccsd(t)")):
        return calc_ccsd_t
    if lname == "fci":
        return calc_fci
    raise ValueError(f"Unknown calculation method: {name}")

def solve_scf(mol, **scfargs):
    HFmethod = scf.HF if not _MUST_UNRES else scf.UHF
    mf = HFmethod(mol).set(init_guess_breaksym=True)
    init_dm = mf.get_init_guess()
    # if _MUST_UNRES:
    #     init_dm[1][:2,:2] = 0
    mf.kernel(init_dm)
    if _USE_NEWTON:
        mf = scf.fast_newton(mf)
    return mf

def calc_hf(mol, **scfargs):
    mf = solve_scf(mol, **scfargs)
    if not mf.converged:
        raise RuntimeError("SCF not converged!")
    etot = mf.e_tot
    grad = mf.nuc_grad_method().kernel() if not _NO_FORCE else None
    rdm = mf.make_rdm1() if not _NO_DM else None
    return etot, grad, rdm

def calc_dft(mol, xc="pbe", **scfargs):
    from pyscf import dft
    KSmethod = dft.KS if not _MUST_UNRES else dft.UKS
    mf = KSmethod(mol, xc).run(**scfargs)
    if not mf.converged:
        raise RuntimeError("SCF not converged!")
    etot = mf.e_tot
    if _NO_FORCE or dft.libxc.xc_type(xc) in ('MGGA', 'NLC'):
        grad = None
    else:
        grad = mf.nuc_grad_method().kernel()
    rdm = mf.make_rdm1() if not _NO_DM else None
    return etot, grad, rdm

def calc_mp2(mol, **scfargs):
    import pyscf.mp
    mf = solve_scf(mol, **scfargs)
    if not mf.converged:
        raise RuntimeError("SCF not converged!")
    postmf = pyscf.mp.MP2(mf).run()
    etot = postmf.e_tot
    grad = postmf.nuc_grad_method().kernel() if not _NO_FORCE else None
    return etot, grad, None

def calc_ccsd(mol, **scfargs):
    import pyscf.cc
    mf = solve_scf(mol, **scfargs)
    if not mf.converged:
        raise RuntimeError("SCF not converged!")
    mycc = mf.CCSD().run()
    etot = mycc.e_tot
    grad = mycc.nuc_grad_method().kernel() if not _NO_FORCE else None
    ccdm = np.einsum('...pi,...ij,...qj->...pq', 
        mf.mo_coeff, mycc.make_rdm1(), mf.mo_coeff.conj()) if not _NO_DM else None
    return etot, grad, ccdm

def calc_ccsd_t(mol, **scfargs):
    import pyscf.cc
    mf = solve_scf(mol, **scfargs)
    if not mf.converged:
        raise RuntimeError("SCF not converged!")
    mycc = mf.CCSD().run()
    et_correction = mycc.ccsd_t()
    etot = mycc.e_tot + et_correction
    if _NO_FORCE:
        return etot, None, None
    import pyscf.grad.ccsd_t as ccsd_t_grad
    grad = ccsd_t_grad.Gradients(mycc).kernel()
    return etot, grad, None

def calc_fci(mol, **scfargs):
    import pyscf.fci
    mf = solve_scf(mol, **scfargs)
    if not mf.converged:
        raise RuntimeError("SCF not converged!")
    myci = pyscf.fci.FCI(mf)
    etot, fcivec = myci.kernel()
    rdm = myci.make_rdm1(fcivec, mol.nao, mol.nelec) if not _NO_DM else None
    return etot, None, rdm


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Calculate and save mp2 energy and mo_coeffs for given xyz files.")
    parser.add_argument("files", nargs="+", help="input xyz files")
    parser.add_argument("-d", "--dump-dir", help="dir of dumped files, default is same dir as xyz file")
    parser.add_argument("-v", "--verbose", default=1, type=int, help="output calculation information")
    parser.add_argument("-B", "--basis", default="ccpvdz", type=str, help="basis used to do the calculation")
    parser.add_argument("-C", "--charge", default=0, type=int, help="net charge of the molecule")
    parser.add_argument("-S", "--spin", default=0, type=int, help="net spin of the molecule")
    parser.add_argument("-M", "--method", default="ccsd", help="method used to do the calculation. support MP2, CCSD and CCSD(T)")
    parser.add_argument("-U", "--unrestrict", action="store_true", help="force using unrestricted methods")
    parser.add_argument("-NF", "--no-force", action="store_true", help="do not calculate force")
    parser.add_argument("-ND", "--no-dm", action="store_true", help="do not calculate dm")
    parser.add_argument("-SO", "--newton", action="store_true", help="allow using newton method when scf not converged")
    parser.add_argument("--scf-input", help="yaml file to specify scf arguments")
    args = parser.parse_args()
    
    if args.unrestrict: _MUST_UNRES = True
    if args.no_force: _NO_FORCE = True
    if args.no_dm: _NO_DM = True
    if args.newton: _USE_NEWTON = True

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
        mol = parse_xyz(fn, args.basis, verbose=args.verbose, charge=args.charge, spin=args.spin)
        try:
            res = calculator(mol, **scfargs)
        except RuntimeError as err:
            print(fn, f"failed, {err}")
            continue
        etot, grad, rdm = res
        if args.dump_dir is None:
            dump_dir = os.path.dirname(fn)
        else:
            dump_dir = args.dump_dir
        dump = os.path.join(dump_dir, os.path.splitext(os.path.basename(fn))[0])
        np.save(dump+".energy.npy", [etot])
        if grad is not None:
            force = -grad / BOHR
            np.save(dump+".force.npy", force)
        if rdm is not None:
            np.save(dump+".dm.npy", rdm)
        if args.verbose:
            print(fn, f"done, time = {time.time()-tic}")