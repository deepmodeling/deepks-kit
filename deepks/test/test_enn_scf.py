
import numpy as np
import torch

import e3nn.o3

from pyscf import gto, scf

from deepks.scf.enn.scf import DSCF, BasisInfo, load_basis


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


if __name__ == '__main__':

    test_desc_equiv()
