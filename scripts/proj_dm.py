import numpy as np
from pyscf import gto
import os
import sys
import argparse
import mendeleev
from calc_eig import calc_eig


# aa = 2.0**np.arange(6,-3,-1)
aa = 1.5**np.array([17,13,10,7,5,3,2,1,0,-1,-2,-3])
bb = np.diag(np.ones(aa.size)) - np.diag(np.ones(aa.size-1), k=1)
SHELL = [aa.size] * 3
coef = np.concatenate([aa.reshape(-1,1), bb], axis=1)
BASIS = [[0, *coef.tolist()], [1, *coef.tolist()], [2, *coef.tolist()]]


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


def gen_proj(mol, intor = 'ovlp', verbose = False) :
    natm = mol.natm
    mole_coords = mol.atom_coords(unit="Ang")
    test_mol = gto.Mole()
    if verbose :
        test_mol.verbose = 4
    else :
        test_mol.verbose = 0
    test_mol.atom = [["Ne", coord] for coord in mole_coords]
    test_mol.basis = BASIS
    test_mol.spin = 0
    test_mol.build(0,0,unit="Ang")
    proj = gto.intor_cross(f'int1e_{intor}_sph', mol, test_mol) 
    
    def proj_func(mo):
        proj_coeff = np.matmul(mo, proj).reshape(*mo.shape[:2], natm, -1)
        if verbose:
            print('shape of coeff data          ', proj_coeff.shape)
        # res : nframe x nocc/nvir x natm x nproj
        return proj_coeff, proj_coeff.shape[-1]
    
    return proj_func


def proj_frame(xyz_file, mo_dir, dump_dir=None, basis='ccpvtz', ename="e_hf", intor='ovlp', verbose=False):
    mol = parse_xyz(xyz_file, basis=basis)
    meta, ehf, e_occ, c_occ = load_data(mo_dir, ename)
    
    proj_func = gen_proj(mol, intor, verbose)
    c_proj_occ,nproj = proj_func(c_occ)
    c_occ = c_proj_occ
    meta = np.append(meta, nproj)
    # print(meta, c_proj_occ.shape)

    if dump_dir is not None:
        dump_data(dump_dir, meta, ehf, e_occ, c_occ)
    return meta, ehf, e_occ, c_occ


def load_data(dir_name, ename="e_hf"):
    meta = np.loadtxt(os.path.join(dir_name, 'system.raw'), dtype=int).reshape(-1)
    natm = meta[0]
    nao = meta[1]
    nocc = meta[2]
    nvir = meta[3]
    ehf = np.loadtxt(os.path.join(dir_name, f'{ename}.raw')).reshape(-1, 1)
    e_occ = np.loadtxt(os.path.join(dir_name, 'ener_occ.raw')).reshape(-1, nocc)
    c_occ = np.loadtxt(os.path.join(dir_name, 'coeff_occ.raw')).reshape([-1, nocc, nao])
    return meta, ehf, e_occ, c_occ


def dump_data(dir_name, meta, ehf, e_occ, c_occ, dm_dict={}) :
    os.makedirs(dir_name, exist_ok = True)
    np.savetxt(os.path.join(dir_name, 'system.raw'), 
               meta.reshape(1,-1), 
               fmt = '%d',
               header = 'natm nao nocc nvir nproj')
    nframe = e_occ.shape[0]
    natm = meta[0]
    nao = meta[1]
    nocc = meta[2]
    nvir = meta[3]
    nproj = meta[4]
    # ntest == natm
    assert(all(c_occ.shape == np.array([nframe, nocc, natm, nproj], dtype=int)))
    assert(all(e_occ.shape == np.array([nframe, nocc], dtype=int)))
    assert(all(all(dm.shape == np.array([nframe, natm, nproj], dtype=int)) for dm in dm_dict.values()))
    np.save(os.path.join(dir_name, 'e_hf.npy'), ehf) 
    np.save(os.path.join(dir_name, 'ener_occ.npy'), e_occ)
    np.save(os.path.join(dir_name, 'coeff_occ.npy'), c_occ)
    for name, dm in dm_dict.items():
        np.save(os.path.join(dir_name, f'{name}.npy'), dm)


def main(xyz_files, mo_dirs, dump_dir, basis='ccpvtz', ename="e_hf", eig_names=['dm_eig', 'od_eig', 'se_eig', 'fe_eig'], intor='ovlp', verbose='False'):
    assert len(xyz_files) == len(mo_dirs)
    oldmeta = None
    all_e_hf = []
    all_e_occ = []
    all_c_occ = []
    all_dm_dict = {name:[] for name in eig_names}
    
    for xf, md in zip(xyz_files, mo_dirs):
        meta, e_hf, e_occ, c_occ = proj_frame(xf, md, basis=basis, ename=ename, intor=intor, verbose=verbose)
        if oldmeta is not None:
            assert all(oldmeta == meta), "all frames has to be in the same system thus meta has to be equal!"
        oldmeta = meta
        all_e_hf.append(e_hf)
        all_e_occ.append(e_occ)
        all_c_occ.append(c_occ)
        for name, dm_list in all_dm_dict.items():
            dm_list.append(2 * calc_eig(name, c_occ, e_occ, xf, shell=SHELL)) # multiply by 2 for restricted method, doubly occupied orbitals
        print(f"{xf} && {md} finished")

    all_e_hf = np.concatenate(all_e_hf)
    all_e_occ = np.concatenate(all_e_occ)
    all_c_occ = np.concatenate(all_c_occ)
    for name in all_dm_dict.keys():
        all_dm_dict[name] = np.concatenate(all_dm_dict[name])

    dump_data(dump_dir, meta, all_e_hf, all_e_occ, all_c_occ, all_dm_dict)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="project mo_coeffs into atomic basis and calculate descriptors.")
    parser.add_argument("-x", "--xyz-file", nargs="+", help="input xyz file(s), if more than one, concat them")
    parser.add_argument("-f", "--mo-dir", nargs="+", help="input mo folder(s), must of same number with xyz files")
    parser.add_argument("-d", "--dump-dir", default=".", help="dir of dumped files, if not specified, use current folder")
    parser.add_argument("-v", "--verbose", action='store_true', help="output calculation information")
    parser.add_argument("-I", "--intor", default="ovlp", help="intor string used to calculate int1e")
    parser.add_argument("-B", "--basis", default="ccpvtz", type=str, help="basis used to do the calculation")
    parser.add_argument("-e", "--ename", default="e_hf", help="file name for total energy")
    parser.add_argument("-E", "--eig-name", nargs="*", default=['dm_eig', 'od_eig', 'se_eig', 'fe_eig'], 
                        help="name of eigen values to be calculated and dumped")
    args = parser.parse_args()
    
    main(args.xyz_file, args.mo_dir, args.dump_dir, args.basis,
         args.ename, args.eig_name, args.intor, args.verbose)