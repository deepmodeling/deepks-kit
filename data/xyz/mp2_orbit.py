# coding: utf-8

import numpy as np
from pyscf import gto, scf, mp
from time import time
import os
import sys
import argparse


def myump2(mf):
    import numpy
    from pyscf import ao2mo
    # As UHF objects, mo_energy, mo_occ, mo_coeff are two-item lists
    # (the first item for alpha spin, the second for beta spin).
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    # mo_coeff_ = mf.mo_coeff
    # mo_coeff= (fix_gauge(mo_coeff_[0]), fix_gauge(mo_coeff_[1]))
    o = numpy.hstack((mo_coeff[0][:,mo_occ[0]>0] ,mo_coeff[1][:,mo_occ[1]>0]))
    # print(mo_occ.shape, mo_occ)
    # print(mo_coeff[0].shape, mo_coeff[1].shape)
    # print(mo_energy[0].shape, mo_energy[1].shape)
    v = numpy.hstack((mo_coeff[0][:,mo_occ[0]==0],mo_coeff[1][:,mo_occ[1]==0]))
    eo = numpy.hstack((mo_energy[0][mo_occ[0]>0] ,mo_energy[1][mo_occ[1]>0]))
    ev = numpy.hstack((mo_energy[0][mo_occ[0]==0],mo_energy[1][mo_occ[1]==0]))
    no = o.shape[1]
    nv = v.shape[1]
    # print(o.shape, no, nv)
    noa = sum(mo_occ[0]>0)
    nva = sum(mo_occ[0]==0)
    eri = ao2mo.general(mf.mol, (o,v,o,v)).reshape(no,nv,no,nv)
    eri[:noa,nva:] = eri[noa:,:nva] = eri[:,:,:noa,nva:] = eri[:,:,noa:,:nva] = 0
    g = eri - eri.transpose(0,3,2,1)
    eov = eo.reshape(-1,1) - ev.reshape(-1)
    de = 1/(eov.reshape(-1,1) + eov.reshape(-1)).reshape(g.shape)
    emp2_ij = .25 * numpy.einsum('iajb,iajb,iajb->ij', g, g, de)
    assert emp2_ij.shape == (no,no)
    emp2 = np.sum(emp2_ij)
    return emp2, emp2_ij


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


def mol_electron(mol, chkfile=None, verbose=False) :
    if verbose:
        start_t = time()
    nao = mol.nao
    natm = mol.natm
    uhf = scf.UHF(mol)
    if chkfile:
        uhf.set(chkfile=chkfile)
    euhf = uhf.kernel()
    if verbose:
        uhf_t = time()
        print(f"time of uhf: {uhf_t - start_t}")

    mo_energy = uhf.mo_energy
    mo_occ = uhf.mo_occ
    # mo_coeff = uhf.mo_coeff
    mo_coeff_ = uhf.mo_coeff
    mo_coeff= (fix_gauge(mo_coeff_[0]), fix_gauge(mo_coeff_[1]))
    occ_a = (mo_occ[0]>0)
    occ_b = (mo_occ[1]>0)
    vir_a = (mo_occ[0]==0)
    vir_b = (mo_occ[1]==0)
    nocc_a = sum(occ_a)
    nocc_b = sum(occ_b)
    nocc = nocc_a + nocc_b
    nvir_a = sum(vir_a)
    nvir_b = sum(vir_b)
    nvir = nvir_a + nvir_b
    assert(nocc + nvir == 2 * nao)
    if verbose :
        print('nao = %d, nocc = %d, nvir = %d, nocc_a = %d, nocc_b = %d' % \
              (nao, nocc, nvir, nocc_a, nocc_b))
        print('shape of a and b coeffs:     ', mo_coeff[0].shape, mo_coeff[1].shape)
    c_occ = np.hstack((mo_coeff[0][:,occ_a], mo_coeff[1][:,occ_b]))
    c_vir = np.hstack((mo_coeff[0][:,vir_a], mo_coeff[1][:,vir_b]))
    e_occ = np.hstack((mo_energy[0][occ_a], mo_energy[1][occ_b]))
    e_vir = np.hstack((mo_energy[0][vir_a], mo_energy[1][vir_b]))
    c_occ = c_occ.T
    c_vir = c_vir.T
    meta = [natm, nao, nocc, nvir]        
    if verbose :
        print('shape of coeff data          ', c_occ.shape)
        print('shape of ener  data          ', e_occ.shape)
        print('shape of coeff data          ', c_vir.shape)
        print('shape of ener  data          ', e_vir.shape)
        mid_t = time()
        # print(f"time of collecting results: {mid_t - uhf_t}")

    # mp2 = mp.MP2(uhf)
    # emp2 = mp2.kernel()
    emp2, emp2_ij = myump2(uhf)
    if verbose :
        print('E(HF)   = %.9g' % euhf)
        print('E(UMP2) = %.9g' % emp2[0])
        print(f"time of mp2: {time()-mid_t}")
    return meta, euhf, emp2, emp2_ij, (e_occ, e_vir), (c_occ, c_vir)
    # return euhf, myemp2, ener_data, coeff_data

    
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


def gen_frame(xyz_file, dump_dir=None, verbose=False):
    if dump_dir is None:
        dump_dir = os.path.splitext(xyz_file)[0]
    mol = parse_xyz(xyz_file, verbose=verbose)
    mol_meta, ehf, emp2, ec_ij, e_data, c_data = mol_electron(mol, verbose=verbose)
    dump_data(dump_dir, mol_meta, ehf, emp2, ec_ij, e_data, c_data)


def main():
    parser = argparse.ArgumentParser(description="Calculate and save mp2 energy and mo_coeffs for given xyz files.")
    parser.add_argument("files", nargs="+", help="input xyz files")
    parser.add_argument("-d", "--dump-dir", default=None, help="dir of dumped files, if not specified, using same dir as input")
    parser.add_argument("-v", "--verbose", action='store_true', help="output calculation information")
    args = parser.parse_args()

    for fn in args.files:
        if args.dump_dir is None:
            dump = None
        else:
            dump = os.path.join(args.dump_dir, os.path.splitext(os.path.basename(fn))[0])
        try:
            gen_frame(fn, dump, args.verbose)
            print(f"{fn} finished")
        except Exception as e:
            print(f"{fn} failed,", e, file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
    