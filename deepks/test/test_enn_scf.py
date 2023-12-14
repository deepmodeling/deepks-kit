
import numpy as np
import torch

import e3nn.o3

from pyscf import gto, scf

from deepks.scf.enn.scf import DSCF, BasisInfo, load_basis, t_get_corr, t_flat_pdms_parity
from deepks.model.model_enn import CorrNet


def make_molecule(atoms, basis, unit='A', verbose=0):

    """
    :param atoms: [Z, x, y, z] of a single molecule
    :param basis: basis of atoms
    :param verbose: int
    :return: mol: gto.Mole
    """

    mol_atom = [[int(x[0]), x[1], x[2], x[3]] for x in atoms]
    nelec = [x[0] for x in mol_atom]

    mol = gto.Mole(
        atom=mol_atom,
        basis=basis,
        unit=unit,
        verbose=verbose,
        spin=int(sum(nelec) % 2)
    )
    mol.build()

    return mol


def make_dm(mol):

    mf = scf.HF(mol).set()
    init_dm = mf.get_init_guess()

    return init_dm


def test_desc_equiv():

    atom_file = '../../examples/water_single/systems/group.03/atom.npy'

    nframe = 5
    atoms = np.load(atom_file)
    atoms = atoms[:nframe]
    basis = 'ccpvdz'
    proj_basis = load_basis(None)

    torch.set_default_dtype(torch.float64)

    # this specifies the change of basis yzx -> xyz
    change_of_coord = torch.tensor([
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.]
    ])
    rot_yzx = e3nn.o3.rand_matrix()  # YXY convention, rotate yzx
    rot_xyz = torch.einsum("ac,cd,db->ab", change_of_coord, rot_yzx, change_of_coord.T)

    basis_info = BasisInfo(proj_basis)

    atoms_rot = atoms[:, :, 1:] @ rot_xyz.numpy().T
    atoms_rot = np.concatenate((atoms[:, :, :1], atoms_rot), axis=-1)
    D_mat = e3nn.o3.Irreps(basis_info.basis_irreps).D_from_matrix(rot_yzx)

    desc = []
    for i in range(nframe):
        mol = make_molecule(atoms[i], basis, 'bohr')
        dm = make_dm(mol)
        dscf = DSCF(mol, None)
        flat_dm = dscf.make_flat_pdm(dm)
        desc.append(torch.from_numpy(flat_dm).double())
    desc = torch.stack(desc, dim=0)

    desc_rot = []
    for i in range(nframe):
        mol = make_molecule(atoms_rot[i], basis, 'bohr')
        dm = make_dm(mol)
        dscf = DSCF(mol, None)
        flat_dm = dscf.make_flat_pdm(dm)
        desc_rot.append(torch.from_numpy(flat_dm).double())
    desc_rot = torch.stack(desc_rot, dim=0)

    torch.testing.assert_close(desc_rot, desc @ D_mat.T)


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


def test_finite_diff():

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    proj_basis = load_basis(None)
    basis_info = BasisInfo(proj_basis)
    model = CorrNet(irreps_in=basis_info.basis_irreps)  # no prefit or preshift

    atom_file = '../../examples/water_single/systems/group.03/atom.npy'
    atoms = np.load(atom_file)
    atoms = atoms[0]
    basis = 'ccpvdz'
    mol = make_molecule(atoms, basis, unit='bohr')
    dscf = DSCF(mol, model)
    dm = make_dm(mol)

    def get_energy(x):
        t_dm = torch.from_numpy(x).double()
        ceig = t_flat_pdms_parity(t_dm, dscf.t_proj_ovlp, basis_info, dscf.cg)
        ec = model(ceig).cpu().detach().numpy()
        return ec

    v_finite = finite_difference(get_energy, dm)
    #print(v_finite.shape)
    #print(np.linalg.norm(v_finite - np.swapaxes(v_finite, 0, 1)))

    ec, vc = t_get_corr(model, torch.from_numpy(dm).double(), dscf.t_proj_ovlp, basis_info, dscf.cg)
    #print(vc.shape)
    assert torch.linalg.norm(vc - v_finite[..., 0]) < 1e-6


def test_vc_symm():

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    proj_basis = load_basis(None)
    basis_info = BasisInfo(proj_basis, symm=True)
    #print(basis_info.basis_irreps)
    assert e3nn.o3.Irreps(basis_info.basis_irreps).dim == basis_info.l3_dim
    model = CorrNet(irreps_in=basis_info.basis_irreps, actv_type='gate')  # no prefit or preshift

    atom_file = '../../examples/water_single/systems/group.03/atom.npy'
    atoms = np.load(atom_file)
    atoms = atoms[0]
    basis = 'ccpvdz'
    mol = make_molecule(atoms, basis, unit='bohr')
    dscf = DSCF(mol, model)
    dm = make_dm(mol)

    ec, vc = t_get_corr(model, torch.from_numpy(dm).double(), dscf.t_proj_ovlp, basis_info, dscf.cg)
    #print(torch.linalg.norm(vc - torch.swapaxes(vc, -1, -2)))
    assert torch.linalg.norm(vc - torch.swapaxes(vc, -1, -2)) < 1e-12


if __name__ == '__main__':

    test_desc_equiv()
    test_finite_diff()
    test_vc_symm()
