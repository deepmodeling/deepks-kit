import numpy as np
from pyscf import gto
import os
import sys
import argparse


def parse_xyz(filename, basis='ccpvtz', verbose=False):
    with open(filename) as fp:
        natoms = int(fp.readline())
        comments = fp.readline()
        xyz_str = "".join(fp.readlines())
    mol = gto.Mole()
    mol.verbose = 4 if verbose else 0
    mol.atom = xyz_str
    mol.basis  = basis
    try:
        mol.build(0,0,unit="Ang")
    except RuntimeError as e:
        mol.spin = 1
        mol.build(0,0,unit="Ang")
    return mol  


def proj(mol, 
         mo, 
         test_ele_num, 
         test_basis, 
         verbose = False) :
    natm = mol.natm
    mole_coords = mol.atom_coords(unit="Ang")
    res = []
    for ii in range(natm):
        test_mol = gto.Mole()
        if verbose :
            test_mol.verbose = 4
        else :
            test_mol.verbose = 0
        test_mol.atom = '%d %f %f %f' % (test_ele_num,
                                         mole_coords[ii][0],
                                         mole_coords[ii][1],
                                         mole_coords[ii][2])
        test_mol.basis = test_basis
        test_mol.spin = test_ele_num % 2
        test_mol.build(0,0,unit="Ang")
        proj = gto.intor_cross('int1e_ovlp_sph', mol, test_mol)        
        n_proj = proj.shape[1]
        proj_coeff = np.matmul(mo, proj)
        res.append(proj_coeff)
    res = np.array(res)
    if verbose:
        print('shape of coeff data          ', res.shape)
    # res : natm x nframe x nocc/nvir x nproj
    return res, proj_coeff.shape[-1]


def load_data(dir_name):
    meta = np.loadtxt(os.path.join(dir_name, 'system.raw'), dtype=int).reshape(-1)
    natm = meta[0]
    nao = meta[1]
    nocc = meta[2]
    nvir = meta[3]
    ehf = np.loadtxt(os.path.join(dir_name, 'e_hf.raw')).reshape(-1, 1)
    emp2 = np.loadtxt(os.path.join(dir_name, 'e_mp2.raw')).reshape(-1, 1)
    e_data = [np.loadtxt(os.path.join(dir_name, 'ener_occ.raw')).reshape(-1, nocc),
              np.loadtxt(os.path.join(dir_name, 'ener_vir.raw')).reshape(-1, nvir)]
    c_data = [np.loadtxt(os.path.join(dir_name, 'coeff_occ.raw')).reshape([-1, nocc, nao]),
              np.loadtxt(os.path.join(dir_name, 'coeff_vir.raw')).reshape([-1, nvir, nao])]
    return meta, ehf, emp2, e_data, c_data


def dump_data(dir_name, meta, ehf, emp2, e_data, c_data) :
    os.makedirs(dir_name, exist_ok = True)
    np.savetxt(os.path.join(dir_name, 'system.raw'), 
               meta.reshape(1,-1), 
               fmt = '%d',
               header = 'natm nao nocc nvir nproj')
    nframe = e_data[0].shape[0]
    natm = meta[0]
    nao = meta[1]
    nocc = meta[2]
    nvir = meta[3]
    nproj = meta[4]
    # ntest == natm
    assert(all(c_data[0].shape == np.array([nframe, nocc, natm, nproj], dtype = int)))
    assert(all(c_data[1].shape == np.array([nframe, nvir, natm, nproj], dtype = int)))
    assert(all(e_data[0].shape == np.array([nframe, nocc], dtype = int)))
    assert(all(e_data[1].shape == np.array([nframe, nvir], dtype = int)))
    np.savetxt(os.path.join(dir_name, 'e_hf.raw'), np.reshape(ehf, [nframe,1])) 
    np.savetxt(os.path.join(dir_name, 'e_mp2.raw'), np.reshape(emp2, [nframe,1])) 
    np.savetxt(os.path.join(dir_name, 'ener_occ.raw'), e_data[0].reshape([nframe, -1]))
    np.savetxt(os.path.join(dir_name, 'ener_vir.raw'), e_data[1].reshape([nframe, -1]))
    np.savetxt(os.path.join(dir_name, 'coeff_occ.raw'), c_data[0].reshape([nframe, -1]))
    np.savetxt(os.path.join(dir_name, 'coeff_vir.raw'), c_data[1].reshape([nframe, -1]))


def proj_frame(xyz_file, mo_dir, dump_dir=None, test_ele_num=10, test_basis="ccpvtz", verbose=False):
    mol = parse_xyz(xyz_file)
    meta, ehf, emp2, e_data, c_data = load_data(mo_dir)
    c_proj_occ,nproj = proj(mol, c_data[0], test_ele_num, test_basis, verbose)
    c_proj_vir,nproj = proj(mol, c_data[1], test_ele_num, test_basis, verbose)

    # [natm, nframe, nocc, nproj] -> [nframe, nocc, natm, nproj]
    # [natm, nframe, nvir, nproj] -> [nframe, nvir, natm, nproj]
    c_proj_occ = np.transpose(c_proj_occ, [1,2,0,3])
    c_proj_vir = np.transpose(c_proj_vir, [1,2,0,3])   
    c_data = (c_proj_occ, c_proj_vir)
    meta = np.append(meta, nproj)
    # print(meta, c_proj_occ.shape)

    if dump_dir is not None:
        dump_data(dump_dir, meta, ehf, emp2, e_data, c_data)
    return meta, ehf, emp2, e_data, c_data


def main():
    parser = argparse.ArgumentParser(description="Calculate and save mp2 energy and mo_coeffs for given xyz files.")
    parser.add_argument("-x", "--xyz-file", nargs="+", help="input xyz file(s), if more than one, concat them")
    parser.add_argument("-f", "--mo-dir", nargs="+", help="input mo folder(s), must of same number with xyz files")
    parser.add_argument("-d", "--dump-dir", default=".", help="dir of dumped files, if not specified, use current folder")
    parser.add_argument("-v", "--verbose", action='store_true', help="output calculation information")
    parser.add_argument("-E", "--ele-num", type=int, default=10, help="element number to use as test orbitals")
    parser.add_argument("-B", "--basis", default="ccpvtz", help="atom basis to use as test orbitals")
    args = parser.parse_args()

    assert len(args.xyz_file) == len(args.mo_dir)
    oldmeta = None
    all_e_hf = []
    all_e_mp2 = []
    all_e_occ = []
    all_e_vir = []
    all_c_occ = []
    all_c_vir = []
    for xf, md in zip(args.xyz_file, args.mo_dir):
        meta, e_hf, e_mp2, e_data, c_data = proj_frame(xf, md, test_ele_num=args.ele_num, test_basis=args.basis, verbose=args.verbose)
        if oldmeta is not None:
            assert all(oldmeta == meta), "all frames has to be in the same system thus meta has to be equal!"
        oldmeta = meta
        all_e_hf.append(e_hf)
        all_e_mp2.append(e_mp2)
        all_e_occ.append(e_data[0])
        all_e_vir.append(e_data[1])
        all_c_occ.append(c_data[0])
        all_c_vir.append(c_data[1])
        print(f"{xf} && {md} finished")
    all_e_hf = np.concatenate(all_e_hf)
    all_e_mp2 = np.concatenate(all_e_mp2)
    all_e_occ = np.concatenate(all_e_occ)
    all_e_vir = np.concatenate(all_e_vir)
    all_c_occ = np.concatenate(all_c_occ)
    all_c_vir = np.concatenate(all_c_vir)

    dump_data(args.dump_dir, meta, all_e_hf, all_e_mp2, (all_e_occ, all_e_vir), (all_c_occ, all_c_vir))
    print("done")

if __name__ == "__main__":
    main()
