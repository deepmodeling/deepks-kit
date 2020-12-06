#!/usr/bin/env python
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -t 24:00:00
#SBATCH --mem=8G

import time
import numpy as np
from deepqc.utils import load_yaml
from deepqc.scf.scf import DSCF
from pyscf import gto, lib
try:
    from pyscf.geomopt.berny_solver import optimize
except ImportError:
    from pyscf.geomopt.geometric_solver import optimize


def run_optim(mol, model=None, proj_basis=None, scf_args={}, conv_args={}):
    cf = DSCF(mol, model, proj_basis=proj_basis).set(**scf_args)
    mol_eq = optimize(cf, **conv_args)
    return mol_eq

def dump_xyz(filename, mol):
    coords = mol.atom_coords(unit="Angstrom").reshape(-1,3)
    elems = mol.elements
    with open(filename, 'w') as fp:
        fp.write(f"{mol.natm}\n\n")
        for x, e in zip(coords, elems):
            fp.write("%s %.18g %.18g %.18g\n" % (e, x[0], x[1], x[2]))


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Calculate and save mp2 energy and mo_coeffs for given xyz files.")
    parser.add_argument("files", nargs="+", help="input xyz files")
    parser.add_argument("-m", "--model-file", help="file of the trained model")
    parser.add_argument("-d", "--dump-dir", help="dir of dumped files, default is same dir as xyz file")
    parser.add_argument("-B", "--basis", default="ccpvdz", type=str, help="basis used to do the calculation")
    parser.add_argument("-P", "--proj_basis", help="basis set used to project dm, must match with model") 
    parser.add_argument("-C", "--charge", default=0, type=int, help="net charge of the molecule")
    parser.add_argument("-v", "--verbose", default=1, type=int, help="output calculation information")
    parser.add_argument("-S", "--suffix", help="suffix added to the saved xyz")
    parser.add_argument("--scf-input", help="yaml file to specify scf arguments")
    parser.add_argument("--conv-input", help="yaml file to specify convergence arguments")
    args = parser.parse_args()
    
    if args.verbose:
        print(f"starting calculation with OMP threads: {lib.num_threads()}",
              f"and max memory: {lib.param.MAX_MEMORY}")

    if args.dump_dir is not None:
        os.makedirs(args.dump_dir, exist_ok = True)
    for fn in args.files:
        tic = time.time()
        mol = gto.M(atom=fn, basis=args.basis, verbose=args.verbose, charge=args.charge, parse_arg=False)
        model = args.model_file
        scf_args = {}
        if args.scf_input is not None:
            argdict = load_yaml(args.scf_input)
            if "scf_args" in argdict:
                scf_args = argdict["scf_args"]
                if model is None and "model" in argdict:
                    model = argdict["model"]
            else:
                scf_args = argdict
        conv_args = load_yaml(args.conv_input) if args.conv_input is not None else {}
        mol_eq = run_optim(mol, model, args.proj_basis, scf_args, conv_args)
        suffix = args.suffix
        if args.dump_dir is None:
            dump_dir = os.path.dirname(fn)
            if not suffix:
                suffix = "eq"
        else:
            dump_dir = args.dump_dir
        dump = os.path.join(dump_dir, os.path.splitext(os.path.basename(fn))[0])
        if suffix:
            dump += f".{suffix}"
        dump_xyz(dump+".xyz", mol_eq)
        if args.verbose:
            print(fn, f"done, time = {time.time()-tic}")