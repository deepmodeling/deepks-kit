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
            factor = np.sign(mo_coeff[jj,ii])
            ret[:,ii] = factor * mo_coeff[:,ii]
            count += 1
    #         break
    # print(count)
    return ret            


def assemble_coeff_matrix(mo_coeff, mo_occ, basis_range, max_nbasis) :
    '''
    give coeff of molecular orbitals, assemble data
    mo_coeff :   coeff of mo
    mo_occ :     occupation of mo. >0 means occ, ==0 meas virtual
    basis_range: range of basis belongs to an atom
    max_nbasis:  maximum number of basis for each atom
    '''    
    natoms = len(basis_range)-1
    occ = (mo_occ>0)
    vir = (mo_occ==0)
    nocc = sum(occ)
    nvir = sum(vir)
    nvec = max_nbasis * natoms * 2      # 2: alpha and beta
    result_occ = np.zeros([nvec, natoms, max_nbasis])
    result_vir = np.zeros([nvec, natoms, max_nbasis])
    for ii in range(natoms) :
        assert(basis_range[ii+1] - basis_range[ii] <= max_nbasis)
        assert(nocc <= nvec)
        assert(nvir <= nvec)
        sub_occ = mo_coeff[basis_range[ii]:basis_range[ii+1],occ].T
        sub_vir = mo_coeff[basis_range[ii]:basis_range[ii+1],vir].T
        result_occ[:sub_occ.shape[0], ii, :sub_occ.shape[1]] = sub_occ
        result_vir[:sub_vir.shape[0], ii, :sub_vir.shape[1]] = sub_vir
    return np.array([result_occ, result_vir])


def assemble_ener_matrix(mo_energy, mo_occ, basis_range, max_nbasis) :
    '''
    give coeff of molecular orbitals, assemble data
    mo_coeff :  coeff of mo
    mo_occ :    occupation of mo. >0 means occ, ==0 meas virtual
    basis_atom: range of basis belongs to an atom
    max_nbasis: maximum number of basis for each atom
    '''    
    natoms = len(basis_range)-1
    occ = (mo_occ>0)
    vir = (mo_occ==0)
    nocc = sum(occ)
    nvir = sum(vir)
    nvec = max_nbasis * natoms * 2
    result_occ = np.zeros([nvec])
    result_vir = np.zeros([nvec])
    for ii in range(natoms) :
        sub_occ = mo_energy[occ]
        sub_vir = mo_energy[vir]
        result_occ[:sub_occ.shape[0]] = sub_occ
        result_vir[:sub_vir.shape[0]] = sub_vir
    return np.array([result_occ, result_vir])


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


def mol_electron(mol, max_basis, verbose = False) :
    nao = mol.nao
    natm = mol.natm
    uhf = scf.UHF(mol)
    euhf = uhf.kernel()
    mo_energy = uhf.mo_energy
    mo_occ = uhf.mo_occ
    # mo_coeff = uhf.mo_coeff
    mo_coeff_ = uhf.mo_coeff
    mo_coeff= (fix_gauge(mo_coeff_[0]), fix_gauge(mo_coeff_[1]))
    nocc_a = sum(mo_occ[0] > 0)
    nocc_b = sum(mo_occ[1] > 0)
    nocc = nocc_a + nocc_b
    if verbose :
        print('nao = %d, nocc = %d, nvir = %d, nocc_a = %d, nocc_b = %d' % \
              (nao, nocc, 2*nao - nocc, nocc_a, nocc_b))
        print('shape of a and b coeffs:     ', mo_coeff[0].shape, mo_coeff[1].shape)
    basis_range = make_basis_range(mol)
    coeff_data_a = assemble_coeff_matrix(mo_coeff[0], mo_occ[0], basis_range, max_basis)
    coeff_data_b = assemble_coeff_matrix(mo_coeff[1], mo_occ[1], basis_range, max_basis)
    coeff_data = np.array([coeff_data_a, coeff_data_b])
    ener_data_a = assemble_ener_matrix(mo_energy[0], mo_occ[0], basis_range, max_basis)
    # print(mo_energy[0])
    # print(ener_data_a)
    ener_data_b = assemble_ener_matrix(mo_energy[1], mo_occ[1], basis_range, max_basis)
    ener_data = np.array([ener_data_a, ener_data_b])
    if verbose :
        print('reserved numb basis per atom ', max_basis)
        print('shape of coeff data          ', coeff_data.shape)
        print('shape of ener  data          ', ener_data.shape)
    mp2 = mp.MP2(uhf)
    emp2 = mp2.kernel()
    myemp2 = myump2(uhf)
    if verbose :
        print('E(HF)        = %.9g' % euhf)
        print('E(UMP2)(std) = %.9g' % emp2[0])
        print('E(UMP2)(my ) = %.9g' % myemp2)
    return euhf, emp2[0], ener_data, coeff_data
    # return euhf, myemp2, ener_data, coeff_data


def make_data(mole_anames, 
              mole_coords, 
              mole_basis,
              max_atom_basis, 
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
    ehf, emp2, e_data, c_data = mol_electron(mol, max_atom_basis, verbose)
    return ehf, emp2, e_data, c_data


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
    np.savetxt(os.path.join(dir_name, 'system.raw'), 
               meta, 
               fmt = '%d',
               header = 'a_b o_v neig natm max_nbas')
    nframe = e_data.shape[0]
    os.makedirs(dir_name, exist_ok = True)
    np.savetxt(os.path.join(dir_name, 'dist.raw'), np.reshape(dist, [nframe,1])) 
    np.savetxt(os.path.join(dir_name, 'e_hf.raw'), np.reshape(ehf, [nframe,1])) 
    np.savetxt(os.path.join(dir_name, 'e_mp2.raw'), np.reshape(emp2, [nframe,1])) 
    np.savetxt(os.path.join(dir_name, 'mo_ener.raw'), e_data.reshape([nframe, -1]))
    np.savetxt(os.path.join(dir_name, 'mo_coeff.raw'), c_data.reshape([nframe, -1]))


def gen_alchemy (ele_list, 
                 relposi_list, 
                 max_nbasis,
                 mol_basis = 'cc-pvdz') :
    nele = len(ele_list)
    # make meta data
    natoms = 2
    neig = max_nbasis * natoms * 2      # 2: alpha and beta
    meta = np.array([2, 2, neig, natoms, max_nbasis], dtype = int)
    for ii in range(nele) :
        for jj in range(ii, nele) :
            print('making ' + ele_list[ii] + ' ' + ele_list[jj])
            # this program gen a mole with 2 atoms!
            # make data for system
            ndist = len(relposi_list)
            all_e_hf = np.zeros(ndist)
            all_e_mp2 = np.zeros(ndist)
            all_e_data = []
            all_c_data = []
            for idx,dd in enumerate(relposi_list) :
                e_hf, e_mp2, e_data, c_data \
                    = make_data([ele_list[ii], ele_list[jj]], 
                                [[0,0,0],[0,0,dd]],
                                mol_basis,
                                max_nbasis,
                                verbose = False)
                all_e_hf [idx] = e_hf
                all_e_mp2[idx] = e_mp2
                all_e_data.append(e_data)
                all_c_data.append(c_data)
            all_e_data = np.array(all_e_data)
            all_c_data = np.array(all_c_data)
            dump_data(ele_list[ii], ele_list[jj], 
                      meta,
                      relposi_list,
                      all_e_hf, 
                      all_e_mp2, 
                      all_e_data,
                      all_c_data)
            
def _main() :
    max_basis = get_nao('Ne', 'ccpvdz')
    # gen_alchemy(['H', 'He', 'Li', 'B', 'N', 'F'], np.arange(0.7,2.0,0.02), max_basis)
    gen_alchemy(['H'], np.arange(0.7,2.0,0.02), max_basis)
    # gen_alchemy(['H', 'He'], np.arange(0.7,2.0,0.02), 'Ne', max_nao)
    # gen_alchemy(['H'], np.arange(0.7,2.0,0.02), 'Ne', max_nao)
    # gen_alchemy(['H','He'], np.arange(0.7,2.0,0.02), max_basis)
    # gen_alchemy(['H'], [0.12], max_basis)

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
