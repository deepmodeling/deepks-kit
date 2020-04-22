import time
import torch
import numpy as np
from pyscf import lib
from pyscf.grad import rhf as rhf_grad


class Gradients(rhf_grad.Gradients):
    # all variables and functions start with "t_" are torch related.
    # convention in einsum:
    #   i,j: orbital
    #   a,b: atom
    #   p,q: projected basis on atom
    #   r,s: mol basis in pyscf
    #   x  : space component of gradient
    #   v  : eigen values of projected dm
    """Analytical nuclear gradient for our SCF model"""
    
    def __init__(self, mf):
        super().__init__(mf)
        self.pmol = mf.pmol
        # < mol_ao | alpha^I_rlm > by shells
        self.t_ovlp_shells = mf.t_ovlp_shells
        # \partial E / \partial (D^I_rl)_mm' by shells
        self.t_gedm_shells = t_get_grad_dms(mf)
        # < \nabla mol_ao | alpha^I_rlm >
        self.t_proj_ipovlp = torch.from_numpy(
            mf.proj_intor("int1e_ipovlp")).double().to(mf.device)
        self._keys.update(self.__dict__.keys())

    def extra_force(self, atom_id, envs):
        """We calculate the pulay force caused by our atomic projection here"""
        de0 = super().extra_force(atom_id, envs)
        dm = envs["dm0"]
        t_dm = torch.from_numpy(dm).double().to(self.base.device)
        t_de = self.t_get_pulay(atom_id, t_dm)
        return de0 + t_de.detach().cpu().numpy()

    def get_hf(self, *args, **kwargs):
        """return the grad given by raw Hartree Fock Hamiltonian under current dm"""
        return rhf_grad.Gradients(self.base).kernel(*args, **kwargs)

    def t_get_pulay(self, atom_id, t_dm):
        """calculate pulay force in torch tensor"""
        # mask to select specifc atom contribution from ipovlp
        mask = self.t_make_mask(atom_id)
        # \partial < mol_ao | aplha^I_rlm' > / \partial X^J
        atom_ipovlp = (self.t_proj_ipovlp * mask).reshape(3, *self.base.t_proj_ovlp.shape)
        # grad X^I w.r.t atomic overlap coeff by shells
        govx_shells = torch.split(atom_ipovlp, self.base.shell_sec, -1)
        # \partial (D^I_rl)_mm' / \partial X^J by shells, lack of symmetrize
        gdmx_shells = [torch.einsum('xrap,rs,saq->xapq', govx, t_dm, po)
                            for govx, po in zip(govx_shells, self.t_ovlp_shells)]
        # \partial E / \partial X^J by shells
        gex_shells = [torch.einsum("xapq,apq->x", gdmx + gdmx.transpose(-1,-2), gedm)
                            for gdmx, gedm in zip(gdmx_shells, self.t_gedm_shells)]
        # total pulay term in gradient
        return torch.stack(gex_shells, 0).sum(0)

    def t_make_mask(self, atom_id):
        mask = torch.from_numpy(
                   make_mask(self.mol, self.pmol, atom_id)
               ).double().to(self.base.device)
        return mask

    def make_grad_pdm_x(self, dm=None, flatten=False):
        if dm is None:
            dm = self.base.make_rdm1()
        t_dm = torch.from_numpy(dm).double().to(self.base.device)
        all_gdmx_shells = self.t_make_grad_pdm_x(t_dm)
        if not flatten:
            return [s.detach().cpu().numpy() for s in all_gdmx_shells]
        else:
            return torch.cat([s.flatten(-2) for s in all_gdmx_shells], 
                             dim=-1).detach().cpu().numpy()

    def t_make_grad_pdm_x(self, t_dm):
        atom_gdmx_shells = []
        for atom_id in range(self.mol.natm):
            mask = self.t_make_mask(atom_id)
            atom_ipovlp = (self.t_proj_ipovlp * mask).reshape(3, *self.base.t_proj_ovlp.shape)
            govx_shells = torch.split(atom_ipovlp, self.base.shell_sec, -1)
            gdmx_shells = [torch.einsum('xrap,rs,saq->xapq', govx, t_dm, po)
                                for govx, po in zip(govx_shells, self.t_ovlp_shells)]
            atom_gdmx_shells.append([gdmx + gdmx.transpose(-1,-2) for gdmx in gdmx_shells])
        # [natom (deriv atom) x 3 (xyz) x natom (proj atom) x nsph (1|3|5) x nsph] list
        all_gdmx_shells = [torch.stack(s, dim=0) for s in zip(*atom_gdmx_shells)]
        return all_gdmx_shells

    def make_grad_eig_x(self, dm=None):
        if dm is None:
            dm = self.base.make_rdm1()
        t_dm = torch.from_numpy(dm).double().to(self.base.device)
        return self.t_make_grad_eig_x(t_dm).detach().cpu().numpy()

    def t_make_grad_eig_x(self, t_dm):
        "v stands for "
        shell_pdm = [torch.einsum('rap,rs,saq->apq', po, t_dm, po).requires_grad_(True)
                        for po in self.t_ovlp_shells]
        calc_eig = lambda dm: torch.symeig(dm, True)[0]
        shell_gvdm = [get_batch_jacobian(calc_eig, dm, dm.shape[-1]) 
                        for dm in shell_pdm]
        shell_gdmx = self.t_make_grad_pdm_x(t_dm)
        shell_gvx = [torch.einsum("bxapq,avpq->bxav", gdmx, gvdm) 
                        for gdmx, gvdm in zip(shell_gdmx, shell_gvdm)]
        return torch.cat(shell_gvx, dim=-1)


def make_mask(mol1, mol2, atom_id):
    mask = np.zeros((mol1.nao, mol2.nao))
    bg1, ed1 = mol1.aoslice_by_atom()[atom_id, 2:]
    bg2, ed2 = mol2.aoslice_by_atom()[atom_id, 2:]
    mask[bg1:ed1, :] -= 1
    mask[:, bg2:ed2] += 1
    return mask


def t_get_grad_dms(mf, dm=None):
    # calculate \partial E / \partial (D^I_rl)_mm' by shells
    if dm is None:
        dm = mf.make_rdm1()
    t_dm = torch.from_numpy(dm).double().to(mf.device)
    proj_dms = [torch.einsum('rap,rs,saq->apq', po, t_dm, po).requires_grad_(True)
                    for po in mf.t_ovlp_shells]
    proj_eigs = [torch.symeig(dm, eigenvectors=True)[0]
                    for dm in proj_dms]
    ceig = torch.cat(proj_eigs, dim=-1).unsqueeze(0) # 1 x natoms x nproj
    ec = mf.net(ceig)
    grad_dms = torch.autograd.grad(ec, proj_dms)
    return grad_dms


def get_batch_jacobian(f, x, noutputs):
    nindim = len(x.shape)-1
    x = x.unsqueeze(1) # b, 1 ,*in_dim
    n = x.shape[0]
    x = x.repeat(1, noutputs, *[1]*nindim) # b, out_dim, *in_dim
    x.requires_grad_(True)
    y = f(x)
    input_val = torch.eye(noutputs).reshape(1,noutputs, noutputs).repeat(n, 1, 1)
    return torch.autograd.grad(y, x, input_val)[0]


# only for testing purpose, not used in code
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


Grad = Gradients

from deepqc.scf.scf import DeepSCF
# Inject to SCF class
DeepSCF.Gradients = lib.class_as_method(Gradients)