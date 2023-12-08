
import torch
import numpy as np

from pyscf import gto
from pyscf import scf, dft

from deepks.utils import load_basis
from deepks.model.model_enn import CorrNet 
from deepks.scf.scf import CorrMixin, PenaltyMixin, gen_proj_mol
from deepks.scf.enn.basis_info import BasisInfo
from deepks.scf.enn.clebsch_gordan import ClebschGordan

DEVICE = 'cpu'


def t_make_pdm(t_dm: torch.Tensor, t_proj: torch.Tensor):

    t_pdm = torch.einsum('...rap,...rs,...saq->...apq', t_proj, t_dm, t_proj)  # (., natom, nproj, nproj)

    return t_pdm


def t_flat_pdms_parity(t_dm: torch.Tensor, t_proj: torch.Tensor,
                       basis_info: BasisInfo, cg_coeffs: ClebschGordan) -> torch.Tensor:

    t_pdm = t_make_pdm(t_dm, t_proj)
    t_ls = basis_info.basis_ls
    t_nls = basis_info.basis_nls
    t_mat_idx = basis_info.basis_mat_idx
    max_l = max(t_ls)

    t_tmp_l_list = [[] for _ in range(2*(2*max_l+1))]
    for idx1, l1 in enumerate(t_ls):
        nl1 = t_nls[idx1]
        for idx2, l2 in enumerate(t_ls):
            nl2 = t_nls[idx2]
            parity = int((-1)**(l1+l2) < 0)
            t_mat = t_pdm[..., t_mat_idx[l1]:t_mat_idx[l1+1], t_mat_idx[l2]:t_mat_idx[l2+1]]
            t_mat = t_mat.reshape((*t_mat.shape[:-2], nl1, 2*l1+1, nl2, 2*l2+1))
            for l3 in range(abs(l1-l2), l1+l2+1):
                cg = cg_coeffs(l1, l2, l3)
                t_trans_mat = torch.einsum('...aibj,ijk->...abk', t_mat, cg)  # (nframe, n_atoms, nl1, nl2, 2*l3+1)
                l3_idx = 2 * l3 + parity
                t_tmp_l_list[l3_idx].append(t_trans_mat.reshape((*t_trans_mat.shape[:-3], nl1*nl2, 2*l3+1)))
    t_l_list = []
    for a in range(2*(2*max_l+1)):
        if len(t_tmp_l_list[a]) > 0:
            t_l_list.append(torch.cat(t_tmp_l_list[a], dim=-2))

    t_pdms_flatten = torch.cat([a.reshape(*a.shape[:-2], -1) for a in t_l_list], dim=-1)

    return t_pdms_flatten


def t_get_corr(model, t_dm, t_proj, basis_info, cg_coeffs, with_vc=True):
    """return the "correction" energy (and potential) given by a NN model"""
    t_dm.requires_grad_(True)
    ceig = t_flat_pdms_parity(t_dm, t_proj, basis_info, cg_coeffs)  # natoms x nproj
    _dref = next(model.parameters()) if isinstance(model, torch.nn.Module) else DEVICE
    ec = model(ceig.to(_dref))  # no batch dim here, unsqueeze(0) if needed
    if not with_vc:
        return ec.to(ceig)
    [vc] = torch.autograd.grad(ec, t_dm, torch.ones_like(ec))

    return ec.to(ceig), vc


class NetMixin(CorrMixin):
    """Mixin class to add correction term given by a neural network model"""

    def __init__(self, model, proj_basis=None, device=DEVICE):
        # make sure you call this method after the base SCF class init
        # otherwise it would throw an error due to the lack of mol attr
        self.device = device
        if isinstance(model, str):
            model = CorrNet.load(model).double()
        if isinstance(model, torch.nn.Module):
            model = model.to(self.device).eval()
        self.net = model
        # try load basis from model file
        if proj_basis is None:
            proj_basis = getattr(model, "_pbas", None)
        # should be a list here, follow pyscf convention
        self._pbas = load_basis(proj_basis)
        # prepare overlap integrals used in projection
        self.prepare_integrals()
        # total number of projected basis per atom
        self.nproj = self.t_proj_ovlp.shape[-1]

        # -- equivariant set up related
        self.basis_info = BasisInfo(self._pbas)
        # TODO: this might be different for different conventions
        self.cg = ClebschGordan(reorder_p=True, change_l3_basis=False)

    def prepare_integrals(self):
        # a virtual molecule to be projected on
        self._pmol = gen_proj_mol(self.mol, self._pbas)
        # < mol_ao | alpha^I_nlm >, shape=[nao x natom x nproj]
        self.t_proj_ovlp = torch.from_numpy(self.proj_ovlp()).double()

    def get_corr(self, dm=None):
        """return "correction" energy and corresponding potential"""
        if dm is None:
            dm = self.make_rdm1()
        if self.net is None:
            return 0., np.zeros_like(dm)
        dm = np.asanyarray(dm)
        if dm.ndim >= 3 and isinstance(self, scf.uhf.UHF):
            dm = dm.sum(0)
        t_dm = torch.from_numpy(dm).double()
        t_ec, t_vc = t_get_corr(self.net, t_dm, self.t_proj_ovlp, self.basis_info, self.cg, with_vc=True)
        ec = t_ec.item() if t_ec.nelement() == 1 else t_ec.detach().cpu().numpy()
        vc = t_vc.detach().cpu().numpy()
        #ec = ec + self.net.get_elem_const(filter(None, self.mol.atom_charges()))
        return ec, vc

    def make_pdm(self, dm=None):
        if dm is None:
            dm = self.make_rdm1()
        t_dm = torch.from_numpy(dm).double()
        t_pdm = t_make_pdm(t_dm, self.t_proj_ovlp)
        return t_pdm.detach().cpu().numpy()

    def make_flat_pdm(self, dm=None):
        if dm is None:
            dm = self.make_rdm1()
        t_dm = torch.from_numpy(dm).double()
        t_dm_flat = t_flat_pdms_parity(t_dm, self.t_proj_ovlp, self.basis_info, self.cg)
        return t_dm_flat.detach().cpu().numpy()

    # -- below are exactly the same as scf/scf.py

    def nuc_grad_method(self):
        from deepks.scf.grad import build_grad
        return build_grad(self)

    def reset(self, mol=None):
        super().reset(mol)
        self.prepare_integrals()
        return self

    def proj_intor(self, intor):
        """1-electron integrals between origin and projected basis"""
        proj = gto.intor_cross(intor, self.mol, self._pmol)
        return proj

    def proj_ovlp(self):
        """overlap between origin and projected basis, reshaped"""
        nao = self.mol.nao
        natm = self._pmol.natm
        pnao = self._pmol.nao
        proj = self.proj_intor("int1e_ovlp")
        # return shape [nao x natom x nproj]
        return proj.reshape(nao, natm, pnao // natm)


# -- the definition of NetMixin is different from scf/scf.py, others are the same
class DSCF(NetMixin, PenaltyMixin, dft.rks.RKS):
    """Restricted SCF solver for given NN energy model"""

    def __init__(self, mol, model, xc="HF", proj_basis=None, penalties=None, device=DEVICE):
        # base method must be initialized first
        dft.rks.RKS.__init__(self, mol, xc=xc)
        # correction mixin initialization
        NetMixin.__init__(self, model, proj_basis=proj_basis, device=device)
        # penalty term initialization
        PenaltyMixin.__init__(self, penalties=penalties)
        # update keys to avoid pyscf warning
        self._keys.update(self.__dict__.keys())


DeepSCF = RDSCF = DSCF


class UDSCF(NetMixin, PenaltyMixin, dft.uks.UKS):
    """Unrestricted SCF solver for given NN energy model"""

    def __init__(self, mol, model, xc="HF", proj_basis=None, penalties=None, device=DEVICE):
        # base method must be initialized first
        dft.uks.UKS.__init__(self, mol, xc=xc)
        # correction mixin initialization
        NetMixin.__init__(self, model, proj_basis=proj_basis, device=device)
        # penalty term initialization
        PenaltyMixin.__init__(self, penalties=penalties)
        # update keys to avoid pyscf warning
        self._keys.update(self.__dict__.keys())
