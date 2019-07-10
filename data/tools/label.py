#!/bin/env/python3 

import sys,dpdata,mendeleev,timeit

from pyscf import gto
from pyscf import scf
from pyscf import mp

def make_label_mp2 (sys, fidx = 0, basis = 'cc-pvdz', verbose = False) :
    natoms = sys.get_natoms()
    at = sys['atom_types']
    an = sys['atom_names']
    cc = sys['coords'][fidx]    
    words = []
    for ii in range(natoms) :        
        words.append('%s %f %f %f' % ( an[at[ii]], cc[ii][0], cc[ii][1], cc[ii][2] ))
    total_charge = 0
    for ii in range(natoms) :
        total_charge += mendeleev.element(an[at[ii]]).atomic_number    

    mol = gto.Mole()
    if verbose :
        mol.verbose = 5
        mol.output = 'mp2.log'
    else :
        mol.verbose = 0
    mol.atom = ';'.join(words)
    mol.spin = total_charge % 2
    mol.basis = basis
    mol.build(0, 0)
    uhf = scf.UHF(mol)
    rhf = scf.RHF(mol)
    stat = timeit.default_timer()
    print('uhf', uhf.kernel())
    print('rhf', rhf.kernel())
    stop = timeit.default_timer()
    print('hf  time: ', stop - stat)
    stat = timeit.default_timer()
    mp2 = mp.MP2(uhf)
    stop = timeit.default_timer()
    print('E(UMP2)(std) = %.9g' % mp2.kernel()[0])
    print('mp2 time: ', stop - stat)

sys = dpdata.System(sys.argv[1], fmt = 'vasp/poscar')
make_label_mp2(sys)

