import torch
import numpy as np
from pyscf import gto, lib
from pyscf.grad import rks as grad_base


class Gradients(grad_base.Gradients):
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
        # prepare integrals for projection and derivative
        self.prepare_integrals()
        # add a field to memorize the pulay term in ec
        self.dec = None
        self._keys.update(self.__dict__.keys())

    def prepare_integrals(self):
        mf = self.base
        self._pmol = mf._pmol
        # < mol_ao | alpha^I_rlm > by shells
        self._t_ovlp_shells = mf._t_ovlp_shells
        # \partial E / \partial (D^I_rl)_mm' by shells
        self._t_gedm_shells = _t_get_grad_dms(mf) if mf.mo_coeff is not None else None
        # < \nabla mol_ao | alpha^I_rlm >
        self._t_proj_ipovlp = torch.from_numpy(
            mf.proj_intor("int1e_ipovlp")).double().to(mf.device)

    def extra_force(self, atom_id, envs):
        """We calculate the pulay force caused by our atomic projection here"""
        de0 = super().extra_force(atom_id, envs)
        dm = envs["dm0"]
        t_dm = torch.from_numpy(dm).double().to(self.base.device)
        t_dec = self._t_get_pulay(atom_id, t_dm)
        dec = t_dec.detach().cpu().numpy()
        # memorize dec results for calculate hf grad
        if self.dec is None:
            self.dec = np.zeros((len(envs["atmlst"]), 3))
        self.dec[envs["k"]] = dec
        # return summed grads
        return de0 + dec
    
    def kernel(self, *args, **kwargs):
        # do nothing additional to the original one but symmetrizing dec
        # return exact the same thing
        de = super().kernel(*args, **kwargs)
        if self.mol.symmetry:
            self.dec = self.symmetrize(self.dec, self.atmlst)
        return de

    def get_base(self):
        """return the grad given by raw Hartree Fock Hamiltonian under current dm"""
        assert self.de is not None and self.dec is not None
        return self.de - self.dec
        
    def _t_get_pulay(self, atom_id, t_dm):
        """calculate pulay force in torch tensor"""
        if self._t_gedm_shells is None:
            self._t_gedm_shells = _t_get_grad_dms(self.base)
        # mask to select specifc atom contribution from ipovlp
        mask = self._t_make_mask(atom_id)
        # \partial < mol_ao | aplha^I_rlm' > / \partial X^J
        atom_ipovlp = (self._t_proj_ipovlp * mask).reshape(3, self.mol.nao, self.mol.natm, -1)
        # grad X^I w.r.t atomic overlap coeff by shells
        govx_shells = torch.split(atom_ipovlp, self.base._shell_sec, -1)
        # \partial (D^I_rl)_mm' / \partial X^J by shells, lack of symmetrize
        gdmx_shells = [torch.einsum('xrap,rs,saq->xapq', govx, t_dm, po)
                            for govx, po in zip(govx_shells, self._t_ovlp_shells)]
        # \partial E / \partial X^J by shells
        gex_shells = [torch.einsum("xapq,apq->x", gdmx + gdmx.transpose(-1,-2), gedm)
                            for gdmx, gedm in zip(gdmx_shells, self._t_gedm_shells)]
        # total pulay term in gradient
        return torch.stack(gex_shells, 0).sum(0)

    def _t_make_mask(self, atom_id):
        mask = torch.from_numpy(
                   make_mask(self.mol, self._pmol, atom_id)
               ).double().to(self.base.device)
        return mask

    def make_grad_pdm_x(self, dm=None, flatten=False):
        if dm is None:
            dm = self.base.make_rdm1()
        t_dm = torch.from_numpy(dm).double().to(self.base.device)
        all_gdmx_shells = self._t_make_grad_pdm_x(t_dm)
        if not flatten:
            return [s.detach().cpu().numpy() for s in all_gdmx_shells]
        else:
            return torch.cat([s.flatten(-2) for s in all_gdmx_shells], 
                             dim=-1).detach().cpu().numpy()

    def _t_make_grad_pdm_x(self, t_dm):
        atom_gdmx_shells = []
        for atom_id in range(self.mol.natm):
            mask = self._t_make_mask(atom_id)
            atom_ipovlp = (self._t_proj_ipovlp * mask).reshape(3, self.mol.nao, self.mol.natm, -1)
            govx_shells = torch.split(atom_ipovlp, self.base._shell_sec, -1)
            gdmx_shells = [torch.einsum('xrap,rs,saq->xapq', govx, t_dm, po)
                                for govx, po in zip(govx_shells, self._t_ovlp_shells)]
            atom_gdmx_shells.append([gdmx + gdmx.transpose(-1,-2) for gdmx in gdmx_shells])
        # [natom (deriv atom) x 3 (xyz) x natom (proj atom) x nsph (1|3|5) x nsph] list
        all_gdmx_shells = [torch.stack(s, dim=0) for s in zip(*atom_gdmx_shells)]
        return all_gdmx_shells

    def make_grad_eig_x(self, dm=None):
        if dm is None:
            dm = self.base.make_rdm1()
        t_dm = torch.from_numpy(dm).double().to(self.base.device)
        return self._t_make_grad_eig_x(t_dm).detach().cpu().numpy()

    def _t_make_grad_eig_x(self, t_dm):
        # v stands for eigen values
        shell_pdm = [torch.einsum('rap,rs,saq->apq', po, t_dm, po).requires_grad_(True)
                        for po in self._t_ovlp_shells]
        calc_eig = lambda dm: torch.symeig(dm, True)[0]
        shell_gvdm = [get_batch_jacobian(calc_eig, dm, dm.shape[-1]) 
                        for dm in shell_pdm]
        shell_gdmx = self._t_make_grad_pdm_x(t_dm)
        shell_gvx = [torch.einsum("bxapq,avpq->bxav", gdmx, gvdm) 
                        for gdmx, gvdm in zip(shell_gdmx, shell_gvdm)]
        return torch.cat(shell_gvx, dim=-1)

    def as_scanner(self):
        scanner = super().as_scanner()
        # make a new version of call method
        class NewScanner(type(scanner)):
            def __call__(self, mol_or_geom, **kwargs):
                if isinstance(mol_or_geom, gto.Mole):
                    mol = mol_or_geom
                else:
                    mol = self.mol.set_geom_(mol_or_geom, inplace=False)

                mf_scanner = self.base
                e_tot = mf_scanner(mol)
                self.mol = mol

                if getattr(self, 'grids', None):
                    self.grids.reset(mol)
                # adding the following line to refresh integrals
                self.prepare_integrals()
                de = self.kernel(**kwargs)
                return e_tot, de

        # hecking the old scanner's method, bind the new one
        scanner.__class__ = NewScanner
        return scanner


def make_mask(mol1, mol2, atom_id):
    mask = np.zeros((mol1.nao, mol2.nao))
    bg1, ed1 = mol1.aoslice_by_atom()[atom_id, 2:]
    bg2, ed2 = mol2.aoslice_by_atom()[atom_id, 2:]
    mask[bg1:ed1, :] -= 1
    mask[:, bg2:ed2] += 1
    return mask


def _t_get_grad_dms(mf, dm=None):
    # calculate \partial E / \partial (D^I_rl)_mm' by shells
    if dm is None:
        dm = mf.make_rdm1()
    t_dm = torch.from_numpy(dm).double().to(mf.device)
    proj_dms = [torch.einsum('rap,rs,saq->apq', po, t_dm, po).requires_grad_(True)
                    for po in mf._t_ovlp_shells]
    if mf.net is None:
        return [torch.zeros_like(pdm) for pdm in proj_dms]
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

# from deepks.scf.scf import DSCF
# # Inject to SCF class
# DSCF.Gradients = lib.class_as_method(Gradients)
