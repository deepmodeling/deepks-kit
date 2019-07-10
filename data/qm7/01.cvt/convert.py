#!/usr/env/bin python3

import os,dpdata
import numpy as np

# numb charge: [ 0,  1,  6,  7,  8, 16]
#                    H   C   N   O   S

z_map = {
    1: 0, 6 : 1, 7 : 2, 8 : 3, 16 : 4
}
name_map = ['H', 'C', 'N', 'O', 'S']

default_box = np.eye(3) * 20

def _get_one_mol(zline, rline) :
    natoms = np.sum(zline != 0) 
    az = zline[:natoms]
    ar = rline[:natoms*3]
    ar = ar.reshape([natoms, 3])
    idx = np.lexsort([az])
    az = az[idx]
    ar = ar[idx]    
    at = np.zeros(natoms, dtype = int)
    for ii in range(natoms) :
        at[ii] = z_map[az[ii]]
    uat = np.unique(at)
    ntypes = uat.size
    atom_numbs = []
    for ii in range(ntypes) :
        atom_numbs.append(np.sum(at == uat[ii]))
    atom_names = []
    for ii in range(ntypes) :
        atom_names.append(name_map[uat[ii]])
    # print(atom_numbs, atom_names)

    data = {}
    data['atom_names'] = atom_names
    data['atom_numbs'] = atom_numbs
    data['atom_types'] = at
    data['coords'] = np.array([ar])
    data['cells'] = np.array([default_box])
    data['orig'] = np.array([0, 0, 0])
    sys = dpdata.System()
    sys.data = data
    sys.to_vasp_poscar('POSCAR')
    # exit(1)
    return atom_names, atom_numbs

def split_mol () :
    zz = np.loadtxt('Z.txt', dtype = int)
    rr = np.loadtxt('R.txt')
    nmols = zz.shape[0]
    for ii in range(nmols) :
        atom_names, atom_numbs = _get_one_mol(zz[ii], rr[ii])
        words = ['%06d_' % ii]
        for kk,ll in zip(atom_names, atom_numbs) :
            words.append(str(kk))
            words.append(str(ll))
            words.append('_')
        words.pop(-1)        
        dir_name = ''.join(words)
        os.makedirs(dir_name, exist_ok = True)
        f_name = os.path.join(dir_name, 'POSCAR')
        if os.path.exists(f_name) :
            os.remove(f_name)
        os.rename('POSCAR', f_name)
        print(' converted ' + dir_name)

split_mol()
