#!/usr/bin/env python3

import os,sys,argparse
import numpy as np
import mendeleev
from pyscf import gto
from pyscf import scf
from pyscf import mp

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
    emp2 = .25 * numpy.einsum('iajb,iajb,iajb->', g, g, de)
    return emp2


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


def make_basis_range(mol) :
    labels = (mol.ao_labels())
    tmp = np.array([int(ii.split()[0]) for ii in labels], dtype = int)
    nbasis = len(tmp)
    cur_atm = 0
    ii = 0
    basis_range = [0]
    while ii < nbasis :
        if tmp[ii] != cur_atm :
            basis_range.append(ii)
            cur_atm = tmp[ii]
        ii += 1
    return np.array(basis_range + [nbasis], dtype = int)


def mol_electron(mol, verbose = False) :
    nao = mol.nao
    natm = mol.natm
    uhf = scf.UHF(mol)
    euhf = uhf.kernel()
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
    assert(nocc + nvir == natm * nao)
    if verbose :
        print('nao = %d, nocc = %d, nvir = %d, nocc_a = %d, nocc_b = %d' % \
              (nao, nocc, natm*nao - nocc, nocc_a, nocc_b))
        print('shape of a and b coeffs:     ', mo_coeff[0].shape, mo_coeff[1].shape)
    c_occ = np.hstack((mo_coeff[0][:,occ_a], mo_coeff[1][:,occ_b]))
    c_vir = np.hstack((mo_coeff[0][:,vir_a], mo_coeff[1][:,vir_b]))
    e_occ = np.hstack((mo_energy[0][occ_a], mo_energy[1][occ_b]))
    e_vir = np.hstack((mo_energy[0][vir_a], mo_energy[1][vir_b]))
    c_occ = c_occ.T
    c_vir = c_vir.T
    if verbose :
        print('shape of coeff data          ', c_occ.shape)
        print('shape of ener  data          ', e_occ.shape)
        print('shape of coeff data          ', c_vir.shape)
        print('shape of ener  data          ', e_vir.shape)
    meta = [natm, nao, nocc, nvir]
    mp2 = mp.MP2(uhf)
    emp2 = mp2.kernel()
    myemp2 = myump2(uhf)
    if verbose :
        print('E(HF)        = %.9g' % euhf)
        print('E(UMP2)(std) = %.9g' % emp2[0])
        print('E(UMP2)(my ) = %.9g' % myemp2)
    return euhf, emp2[0], (e_occ, e_vir), (c_occ, c_vir), meta
    # return euhf, myemp2, ener_data, coeff_data

def proj(mol, 
         mo, 
         test_name, 
         test_basis, 
         verbose = False) :
    natm = mol.natm
    mole_coords = mol.atom_coords()
    res = []
    for ii in range(natm):
        test_mol = gto.Mole()
        if verbose :
            test_mol.verbose = 5
        else :
            test_mol.verbose = 0
        test_mol.output = 'test.log'
        test_mol.atom = '%s %f %f %f' % (test_name,
                                         mole_coords[ii][0],
                                         mole_coords[ii][1],
                                         mole_coords[ii][2])
        test_mol.basis = test_basis
        test_mol.spin = mendeleev.element(test_name).atomic_number % 2
        test_mol.build(0,0)
        proj = gto.intor_cross('int1e_ovlp_sph', mol, test_mol)        
        n_proj = proj.shape[1]
        proj_coeff = np.matmul(mo, proj)
        res.append(proj_coeff)
    res = np.array(res)
    if verbose:
        print('shape of coeff data          ', res.shape)
    return res, proj_coeff.shape[-1]


def make_data(mole_anames, 
              mole_coords, 
              mole_basis,
              test_name, 
              test_basis,
              verbose = False) :
    # make molecule    
    atom_str_list = []
    atomic_number = []
    for ii,jj in zip(mole_anames, mole_coords) :
        atom_str_list.append(ii + " %f %f %f" % (jj[0], jj[1], jj[2]))
        atomic_number.append(mendeleev.element(ii).atomic_number)
    natoms = len(atom_str_list)
    mol = gto.Mole()
    if verbose :
        mol.verbose = 5
    else :
        mol.verbose = 0
    mol.output = 'mol.log'
    mol.atom = ';'.join(atom_str_list)
    mol.spin = sum(atomic_number) % 2
    mol.basis = mole_basis
    mol.build(0,0)
    # electron structure of mol
    ehf, emp2, e_data, c_data, mol_meta = mol_electron(mol, verbose)
    c_proj_occ,nproj = proj(mol, c_data[0], test_name, test_basis, verbose)
    c_proj_vir,nproj = proj(mol, c_data[1], test_name, test_basis, verbose)
    # [natm, nocc, nproj] -> [nocc, natm, nproj]
    # [natm, nvir, nproj] -> [nvir, natm, nproj]
    c_proj_occ = np.transpose(c_proj_occ, [1,0,2])
    c_proj_vir = np.transpose(c_proj_vir, [1,0,2])
    mol_meta.append(nproj)
    return ehf, emp2, e_data, (c_proj_occ, c_proj_vir), mol_meta


def get_nao (ele, basis) :
    test_mol = gto.Mole()
    test_mol.verbose = 0
    test_mol.atom = '%s 0 0 0' % ele
    test_mol.spin = mendeleev.element(ele).atomic_number % 2
    test_mol.basis = basis
    test_mol.build(0,0)
    return test_mol.nao    


def dump_data(ele1, ele2, meta, dist, ehf, emp2, e_data, c_data) :
    dir_name = 'data_' + ele1 + '_' + ele2
    os.makedirs(dir_name, exist_ok = True)
    np.savetxt(os.path.join(dir_name, 'system.raw'), 
               meta, 
               fmt = '%d',
               header = 'natm nao nocc nvir')
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
    np.savetxt(os.path.join(dir_name, 'dist.raw'), np.reshape(dist, [nframe,1])) 
    np.savetxt(os.path.join(dir_name, 'e_hf.raw'), np.reshape(ehf, [nframe,1])) 
    np.savetxt(os.path.join(dir_name, 'e_mp2.raw'), np.reshape(emp2, [nframe,1])) 
    np.savetxt(os.path.join(dir_name, 'ener_occ.raw'), e_data[0].reshape([nframe, -1]))
    np.savetxt(os.path.join(dir_name, 'ener_vir.raw'), e_data[1].reshape([nframe, -1]))
    np.savetxt(os.path.join(dir_name, 'coeff_occ.raw'), c_data[0].reshape([nframe, -1]))
    np.savetxt(os.path.join(dir_name, 'coeff_vir.raw'), c_data[1].reshape([nframe, -1]))


def gen_alchemy (ele_list, 
                 relposi_list, 
                 mol_basis = 'cc-pvdz', 
                 test_name = 'Ne',
                 test_basis = 'cc-pvdz') :
    nele = len(ele_list)
    # make meta data
    for ii in range(nele) :
        for jj in range(ii, nele) :
            print('making ' + ele_list[ii] + ' ' + ele_list[jj])
            # this program gen a mole with 2 atoms!
            # make data for system
            ndist = len(relposi_list)
            all_e_hf = np.zeros(ndist)
            all_e_mp2 = np.zeros(ndist)
            all_e_occ = []
            all_e_vir = []
            all_c_occ = []
            all_c_vir = []
            for idx,dd in enumerate(relposi_list) :
                e_hf, e_mp2, e_data, c_data, mol_meta\
                    = make_data([ele_list[ii], ele_list[jj]], 
                                [[0,0,0],[0,0,dd]],
                                mol_basis,
                                test_name,
                                test_basis,
                                verbose = False)
                all_e_hf [idx] = e_hf
                all_e_mp2[idx] = e_mp2
                all_e_occ.append(e_data[0])
                all_e_vir.append(e_data[1])
                all_c_occ.append(c_data[0])
                all_c_vir.append(c_data[1])
            all_e_occ = np.array(all_e_occ)
            all_e_vir = np.array(all_e_vir)
            all_c_occ = np.array(all_c_occ)
            all_c_vir = np.array(all_c_vir)
            dump_data(ele_list[ii], ele_list[jj], 
                      mol_meta,
                      relposi_list,
                      all_e_hf, 
                      all_e_mp2, 
                      (all_e_occ, all_e_vir),
                      (all_c_occ, all_c_vir))
            
def _main() :
    gen_alchemy(['H'], 
                np.arange(0.7,2.0,0.02), 
                mol_basis = 'cc-pvdz',
                test_name = 'Ne',
                test_basis = 'cc-pvtz')
    # gen_alchemy(['H', 'He', 'Li', 'B', 'N', 'F'], 
    #             np.arange(0.7,2.0,0.02), 
    #             mol_basis = 'cc-pvdz',
    #             test_name = 'Ne',
    #             test_basis = 'cc-pvtz')
    # gen_alchemy(['H', 'He'], np.arange(0.7,2.0,0.02))
    # gen_alchemy(['H'], [0.12], 'ccpvdz', 'Ne', 'cc-pvtz')

    # e_occ, e_vir, s_occ, s_vir \
    #     = make_data(['F', 'He'], 
    #                 [[0,0,0],[0,1.2,0]],
    #                 'ccpvdz', 
    #                 'Ne',
    #                 'cc-pv5z', 
    #                 verbose = True)
    # print(e_occ.shape, e_vir.shape, s_occ.shape, s_vir.shape)

if __name__ == '__main__' :
    _main()
