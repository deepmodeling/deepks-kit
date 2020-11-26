#!/usr/bin/env python
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -t 24:00:00
#SBATCH --mem=32G

import time
import numpy as np
from deepqc.utils import load_yaml
from deepqc.scf.scf import DSCF
from pyscf import gto, lib

BOHR = 0.52917721092

def finite_difference(f, x, delta=1e-6):
    in_shape = x.shape
    y0 = f(x)
    out_shape = y0.shape
    res = np.empty(in_shape + out_shape)
    for idx in np.ndindex(*in_shape):
        diff = np.zeros(in_shape)
        diff[idx] += delta
        y1 = f(x+diff)
        res[idx] = (y1-y0) / delta
    return res

def calc_deriv(mol, model=None, **scfargs):
    tic = time.time()
    cf = DSCF(mol, model).run(**scfargs)
    if not cf.converged:
        raise RuntimeError("SCF not converged!")
    ff = cf.nuc_grad_method().run()
    if mol.verbose:
        print(f"step time = {time.time()-tic}")
    return ff.de

def make_closure(mol, model=None, **scfargs):
    refmol = mol
    unit = refmol.unit
    def cc2de(coords):
        mol = refmol.set_geom_(coords, inplace=False)
        de = calc_deriv(mol, model, **scfargs)
        if not unit.upper().startswith(("B", "AU")):
            de /= BOHR
        return de
    return cc2de
    # scanner is not very stable. We construct new scf objects every time
    # scanner = DSCF(mol, model).set(**scfargs).nuc_grad_method().as_scanner()
    # return lambda m: scanner(m)[-1]

def calc_hessian(mol, model=None, delta=1e-6, **scfargs):
    cc2de = make_closure(mol, model, **scfargs)
    cc0 = mol.atom_coords(unit=mol.unit)
    hess = finite_difference(cc2de, cc0, delta).transpose((0,2,1,3))
    return hess


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Calculate and save mp2 energy and mo_coeffs for given xyz files.")
    parser.add_argument("files", nargs="+", help="input xyz files")
    parser.add_argument("-m", "--model-file", help="file of the trained model")
    parser.add_argument("-d", "--dump-dir", help="dir of dumped files, default is same dir as xyz file")
    parser.add_argument("-D", "--delta", default=1e-6, type=float, help="numerical difference step size")
    parser.add_argument("-v", "--verbose", default=0, type=int, help="output calculation information")
    parser.add_argument("-B", "--basis", default="ccpvdz", type=str, help="basis used to do the calculation")
    parser.add_argument("-C", "--charge", default=0, type=int, help="net charge of the molecule")
    parser.add_argument("--bohr", action="store_true", help="use Bohr as length unit")
    parser.add_argument("--scf-input", help="yaml file to specify scf arguments")
    args = parser.parse_args()
    
    if args.verbose:
        print(f"starting calculation with OMP threads: {lib.num_threads()}",
              f"and max memory: {lib.param.MAX_MEMORY}")

    if args.dump_dir is not None:
        os.makedirs(args.dump_dir, exist_ok = True)
    for fn in args.files:
        tic = time.time()
        mol = gto.M(atom=fn, basis=args.basis, verbose=args.verbose, charge=args.charge, parse_arg=False)
        if args.bohr:
            mol.set(unit="Bohr")
        model = args.model_file
        scfargs = {}
        if args.scf_input is not None:
            argdict = load_yaml(args.scf_input)
            if "scf_args" in argdict:
                scfargs = argdict["scf_args"]
                if model is None and "model" in argdict:
                    model = argdict["model"]
            else:
                scfargs = argdict
        hess = calc_hessian(mol, model, args.delta, **scfargs)
        if args.dump_dir is None:
            dump_dir = os.path.dirname(fn)
        else:
            dump_dir = args.dump_dir
        dump = os.path.join(dump_dir, os.path.splitext(os.path.basename(fn))[0])
        np.save(dump+".hessian.npy", hess)
        print(fn, f"done, time = {time.time()-tic}")