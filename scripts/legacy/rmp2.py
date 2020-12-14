# coding: utf-8

import numpy as np
from pyscf import gto, scf, mp, lib
from pyscf.mp.mp2 import _mo_energy_without_core
from time import time
import os
import sys
import argparse


def my_kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_eij=True):
    if mo_energy is None or mo_coeff is None:
        if mp.mo_energy is None or mp.mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call mf.kernel() to generate them.')
        mo_coeff = None
        mo_energy = _mo_energy_without_core(mp, mp.mo_energy)
    else:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert(mp.frozen is 0 or mp.frozen is None)

    if eris is None: eris = mp.ao2mo(mo_coeff)

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_eij:
        eij = np.empty((nocc,nocc), dtype=eia.dtype)
    else:
        eij = None

    emp2 = 0
    for i in range(nocc):
        gi = np.asarray(eris.ovov[i*nvir:(i+1)*nvir])
        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        tmp_eij = 2 * np.einsum('jab,jab->j', t2i, gi) - np.einsum('jab,jba->j', t2i, gi)
        emp2 += tmp_eij.sum()
        if with_eij:
            eij[i] = tmp_eij

    return emp2.real, eij.real


def parse_xyz(filename, basis='ccpvtz', verbose=False):
    with open(filename) as fp:
        natoms = int(fp.readline())
        comments = fp.readline()
        xyz_str = "".join(fp.readlines())
    mol = gto.Mole()
    mol.verbose = 4 if verbose else 0
    mol.atom = xyz_str
    mol.basis  = basis
    mol.build(0,0,unit="Ang")
    return mol  


def fix_gauge(mo_coeff) :
    nvec = mo_coeff.shape[1]
    ndim = mo_coeff.shape[0]
    ret = np.zeros(mo_coeff.shape)
    count = 0
    for ii in range(nvec) :
        for jj in range(ndim) :
            if np.sign(mo_coeff[jj,ii]) != 0 :
                break
        if jj == ndim :
            # mo_coeff[:,ii] == 0
            assert(np.max(np.abs(mo_coeff[:,ii])) == 0)
            raise RuntimeError( 'ERROR: zero eigen func, should not happen')
            continue
        else :
            if (jj != 0) :
                print('gauge ref is not 0')
            factor = np.sign(mo_coeff[jj,ii])
            ret[:,ii] = factor * mo_coeff[:,ii]
            count += 1
    #         break
    # print(count)
    return ret


def mol_electron(mol, frozen=0, chkfile=None, verbose=False) :
    if verbose:
        start_t = time()
    nao = mol.nao
    natm = mol.natm
    rhf = scf.RHF(mol)
    if chkfile:
        rhf.set(chkfile=chkfile)
    erhf = rhf.kernel()
    if verbose:
        rhf_t = time()
        print(f"time of rhf: {rhf_t - start_t}")

    mo_energy = rhf.mo_energy
    mo_occ = rhf.mo_occ
    # mo_coeff = rhf.mo_coeff
    mo_coeff_ = rhf.mo_coeff
    mo_coeff= fix_gauge(mo_coeff_)
    occ_a = (mo_occ>0)
    occ_a[:frozen] = False
    # occ_b = (mo_occ[1]>0)
    vir_a = (mo_occ==0)
    # vir_b = (mo_occ[1]==0)
    nocc_a = sum(occ_a)
    # nocc_b = sum(occ_b)
    nocc = nocc_a
    nvir_a = sum(vir_a)
    # nvir_b = sum(vir_b)
    nvir = nvir_a
    assert(nocc + nvir + frozen == nao)
    if verbose :
        print('nao = %d, nocc = %d, nvir = %d' % \
              (nao, nocc, nvir))
        print('shape of a and b coeffs:     ', mo_coeff[0].shape, mo_coeff[1].shape)
    c_occ = mo_coeff[:,occ_a]
    c_vir = mo_coeff[:,vir_a]
    e_occ = mo_energy[occ_a]
    e_vir = mo_energy[vir_a]
    c_occ = c_occ.T
    c_vir = c_vir.T
    meta = [natm, nao, nocc, nvir]        
    if verbose :
        print('shape of coeff data          ', c_occ.shape)
        print('shape of ener  data          ', e_occ.shape)
        print('shape of coeff data          ', c_vir.shape)
        print('shape of ener  data          ', e_vir.shape)
        mid_t = time()
        # print(f"time of collecting results: {mid_t - rhf_t}")

    mp2 = mp.MP2(rhf, frozen=frozen)
    # emp2 = mp2.kernel()
    emp2, emp2_ij = my_kernel(mp2)
    if verbose :
        print('E(HF)   = %.9g' % erhf)
        print('E(RMP2) = %.9g' % emp2)
        print(f"time of mp2: {time()-mid_t}")
    return meta, erhf, emp2, emp2_ij, (e_occ, e_vir), (c_occ, c_vir)
    # return erhf, myemp2, ener_data, coeff_data

    
def dump_data(dir_name, meta, ehf, emp2, ec_ij, e_data, c_data) :
    os.makedirs(dir_name, exist_ok = True)
    np.savetxt(os.path.join(dir_name, 'system.raw'), 
               np.array(meta).reshape(1,-1), 
               fmt = '%d',
               header = 'natm nao nocc nvir')
    nframe = 1
    natm = meta[0]
    nao = meta[1]
    nocc = meta[2]
    nvir = meta[3]
    # ntest == natm
    assert(all(c_data[0].shape == np.array([nocc, nao], dtype = int)))
    assert(all(c_data[1].shape == np.array([nvir, nao], dtype = int)))
    assert(all(e_data[0].shape == np.array([nocc], dtype = int)))
    assert(all(e_data[1].shape == np.array([nvir], dtype = int)))
    assert(all(ec_ij.shape == np.array([nocc, nocc], dtype = int)))
    np.savetxt(os.path.join(dir_name, 'e_hf.raw'), np.reshape(ehf, [nframe,1])) 
    np.savetxt(os.path.join(dir_name, 'e_mp2.raw'), np.reshape(emp2, [nframe,1])) 
    np.savetxt(os.path.join(dir_name, 'ec_ij.raw'), ec_ij.reshape([nframe, -1]))
    np.savetxt(os.path.join(dir_name, 'ener_occ.raw'), e_data[0].reshape([nframe, -1]))
    np.savetxt(os.path.join(dir_name, 'ener_vir.raw'), e_data[1].reshape([nframe, -1]))
    np.savetxt(os.path.join(dir_name, 'coeff_occ.raw'), c_data[0].reshape([nframe, -1]))
    np.savetxt(os.path.join(dir_name, 'coeff_vir.raw'), c_data[1].reshape([nframe, -1]))


def gen_frame(xyz_file, basis='ccpvtz', frozen=0, dump_dir=None, verbose=False):
    if dump_dir is None:
        dump_dir = os.path.splitext(xyz_file)[0]
    mol = parse_xyz(xyz_file, basis=basis ,verbose=verbose)
    mol_meta, ehf, emp2, ec_ij, e_data, c_data = mol_electron(mol, frozen=frozen, verbose=verbose)
    dump_data(dump_dir, mol_meta, ehf, emp2, ec_ij, e_data, c_data)


def main():
    parser = argparse.ArgumentParser(description="Calculate and save mp2 energy and mo_coeffs for given xyz files.")
    parser.add_argument("files", nargs="+", help="input xyz files")
    parser.add_argument("-d", "--dump-dir", default=None, help="dir of dumped files, if not specified, using same dir as input")
    parser.add_argument("-v", "--verbose", action='store_true', help="output calculation information")
    parser.add_argument("-F", "--frozen", default=0, type=int, help="number of orbit to be frozen when calculate mp2")
    parser.add_argument("-B", "--basis", default="ccpvtz", type=str, help="basis used to do the calculation")
    args = parser.parse_args()

    for fn in args.files:
        if args.dump_dir is None:
            dump = None
        else:
            dump = os.path.join(args.dump_dir, os.path.splitext(os.path.basename(fn))[0])
        try:
            gen_frame(fn, args.basis, args.frozen, dump, args.verbose)
            print(f"{fn} finished")
        except Exception as e:
            print(f"{fn} failed,", e, file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
