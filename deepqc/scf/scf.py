import time
import torch
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import dft
from deepqc.utils import check_list
from deepqc.train.model import QCNet

DEVICE = 'cpu'#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# all variables and functions start with "t_" are torch based.
# all variables and functions ends with "0" are original base method results
# convention in einsum:
#   i,j: orbital
#   a,b: atom
#   p,q: projected basis on atom
#   r,s: mol basis in pyscf
# parameter shapes:
#   ovlp_shells: [nao x natom x nsph] list
#   pdm_shells: [natom x nsph x nsph] list
#   eig_shells: [natom x nsph] list


def t_make_pdm(dm, ovlp_shells):
    """return projected density matrix by shell"""
    # (D^I_rl)_mm' = \sum_i < alpha^I_rlm | phi_i >< phi_i | aplha^I_rlm' >
    pdm_shells = [torch.einsum('rap,rs,saq->apq', po, dm, po)
                    for po in ovlp_shells]
    return pdm_shells


def t_make_eig(dm, ovlp_shells):
    """return eigenvalues of projected density matrix"""
    pdm_shells = t_make_pdm(dm, ovlp_shells)
    eig_shells = [torch.symeig(dm, eigenvectors=True)[0]
                    for dm in pdm_shells]
    ceig = torch.cat(eig_shells, dim=-1)
    return ceig


def t_get_corr(model, dm, ovlp_shells, with_vc=True):
    """return the "correction" energy (and potential) given by a NN model"""
    dm.requires_grad_(True)
    ceig = t_make_eig(dm, ovlp_shells) # natoms x nproj
    _dref = next(model.parameters())
    ec = model(ceig.to(_dref))  # no batch dim here, unsqueeze(0) if needed
    if not with_vc:
        return ec
    [vc] = torch.autograd.grad(ec, dm)
    return ec.to(ceig), vc


class DSCF(dft.rks.RKS):
    """Self Consistant Field solver for given QC model"""
    
    def __init__(self, mol, model, xc="HF", proj_basis=None, penalties=None, device=DEVICE):
        super().__init__(mol, xc=xc)
        self.device = device
        if isinstance(model, str):
            model = QCNet.load(model).double()
        if isinstance(model, torch.nn.Module):
            model = model.to(self.device)
        self.net = model

        # should be a list here, follow pyscf convention
        self._pbas = load_basis(proj_basis)
        # [1,1,1,...,3,3,3,...,5,5,5,...]
        self._shell_sec = sum(([2*b[0]+1] * (len(b)-1) for b in self._pbas), [])
        # total number of projected basis per atom
        self.nproj = sum(self._shell_sec)
        # prepare overlap integrals used in projection
        self.prepare_integrals()

        # initialize penalty terms
        self.penalties = check_list(penalties)
        for pnt in self.penalties:
            pnt.init_hook(self)
        # update keys to avoid pyscf warning
        self._keys.update(self.__dict__.keys())

    def prepare_integrals(self):
        # a virtual molecule to be projected on
        self._pmol = gen_proj_mol(self.mol, self._pbas)
        # < mol_ao | alpha^I_rlm >, shape=[nao x natom x nproj]
        t_proj_ovlp = torch.from_numpy(self.proj_ovlp()).double()
        # split the projected coeffs by shell (different r and l)
        self._t_ovlp_shells = torch.split(t_proj_ovlp, self._shell_sec, -1)

    def get_veff0(self, *args, **kwargs):
        return super().get_veff(*args, **kwargs)

    def energy_elec0(self, dm=None, h1e=None, vhf=None):
        if vhf is None: vhf = self.get_veff0(dm=dm)
        return super().energy_elec(dm, h1e, vhf)
    
    def energy_tot0(self, dm=None, h1e=None, vhf=None):
        return self.energy_elec0(dm, h1e, vhf)[0] + self.energy_nuc()

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        """original mean field potential + correction potential"""
        if mol is None: 
            mol = self.mol
        if dm is None: 
            dm = self.make_rdm1()
        tic = (time.clock(), time.time())
        assert isinstance(dm, np.ndarray) and dm.ndim == 2
        
        # base method part
        v0_last = getattr(vhf_last, 'v0', 0)
        v0 = self.get_veff0(mol, dm, dm_last, v0_last, hermi)
        tic = logger.timer(self, 'v0', *tic)
        # Correlation (or correction) part
        ec, vc = self.get_corr(dm)
        tic = logger.timer(self, 'vc', *tic)

        vtot = v0 + vc
        vtot = lib.tag_array(vtot, ec=ec, v0=v0)
        return vtot

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        """return electronic energy and the 2-electron part contribution"""
        if dm is None: 
            dm = self.make_rdm1()
        if h1e is None: 
            h1e = self.get_hcore()
        if vhf is None or getattr(vhf, 'ec', None) is None: 
            vhf = self.get_veff(dm=dm)
        etot, e2 = self.energy_elec0(dm, h1e, vhf.v0)
        ec = vhf.ec
        logger.debug(self, f'Emodel = {ec}')
        return (etot+ec).real, e2+ec
    
    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, 
                 diis=None, diis_start_cycle=None, 
                 level_shift_factor=None, damp_factor=None):
        """modified get_fock method to apply penalty terms onto vhf"""
        if dm is None:
            dm = self.make_rdm1()
        if h1e is None: 
            h1e = self.get_hcore()
        if vhf is None: 
            vhf = self.get_veff(dm=dm)
        vp = sum(pnt.fock_hook(self, dm=dm, h1e=h1e, vhf=vhf, cycle=cycle) 
                    for pnt in self.penalties)
        vhf = vhf + vp
        return super().get_fock(h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=cycle, 
                        diis=diis, diis_start_cycle=diis_start_cycle, 
                        level_shift_factor=level_shift_factor, damp_factor=damp_factor)

    def get_corr(self, dm=None):
        """return "correction" energy and corresponding potential"""
        if dm is None:
            dm = self.make_rdm1()
        if self.net is None:
            return 0., np.zeros_like(dm)
        t_dm = torch.from_numpy(dm).double()
        t_ec, t_vc = t_get_corr(self.net, t_dm, self._t_ovlp_shells, with_vc=True)
        return t_ec.item(), t_vc.detach().cpu().numpy()

    def make_pdm(self, dm=None, flatten=False):
        """return projected density matrix by shell"""
        if dm is None:
            dm = self.make_rdm1()
        t_dm = torch.from_numpy(dm).double()
        t_pdm_shells = t_make_pdm(t_dm, self._t_ovlp_shells)
        if not flatten:
            return [s.detach().cpu().numpy() for s in t_pdm_shells]
        else:
            return torch.cat([s.flatten(-2) for s in t_pdm_shells], 
                             dim=-1).detach().cpu().numpy()

    def make_eig(self, dm=None):
        """return eigenvalues of projected density matrix"""
        if dm is None:
            dm = self.make_rdm1()
        t_dm = torch.from_numpy(dm).double()
        t_eig = t_make_eig(t_dm, self._t_ovlp_shells)
        return t_eig.detach().cpu().numpy()

    def proj_intor(self, intor):
        """1-electron integrals between origin and projected basis"""
        proj = gto.intor_cross(intor, self.mol, self._pmol) 
        return proj
        
    def proj_ovlp(self):
        """overlap between origin and projected basis, reshaped"""
        nao = self.mol.nao
        natm = self.mol.natm
        pnao = self._pmol.nao
        proj = self.proj_intor("int1e_ovlp")
        # return shape [nao x natom x nproj]
        return proj.reshape(nao, natm, pnao // natm)

    def reset(self, mol=None):
        super().reset(mol)
        self.prepare_integrals()
        return self

    def nuc_grad_method(self):
        # if self.net is None:
        #     return super().nuc_grad_method()
        from deepqc.scf.grad import Gradients
        return Gradients(self)

    def nuc_grad_method0(self):
        return super().nuc_grad_method()


DeepSCF = DSCF
RDSCF = DSCF


def gen_proj_mol(mol, basis) :
    natm = mol.natm
    nao = mol.nao
    mole_coords = mol.atom_coords(unit="Ang")
    test_mol = gto.Mole()
    test_mol.atom = [["Ne", coord] for coord in mole_coords]
    test_mol.basis = basis
    test_mol.build(0,0,unit="Ang")
    return test_mol


_zeta = 1.5**np.array([17,13,10,7,5,3,2,1,0,-1,-2,-3])
_coef = np.diag(np.ones(_zeta.size)) - np.diag(np.ones(_zeta.size-1), k=1)
_table = np.concatenate([_zeta.reshape(-1,1), _coef], axis=1)
DEFAULT_BASIS = [[0, *_table.tolist()], [1, *_table.tolist()], [2, *_table.tolist()]]

def load_basis(basis):
    if basis is None:
        return DEFAULT_BASIS
    elif isinstance(basis, np.ndarray) and basis.ndim == 2:
        return [[ll, *basis.tolist()] for ll in range(3)]
    elif not isinstance(basis, str):
        return basis
    elif basis.endswith(".npy"):
        table = np.load(basis)
        return [[ll, *table.tolist()] for ll in range(3)]
    elif basis.endswith(".npz"):
        all_tables = np.load(basis).values()
        return [[ll, *table.tolist()] for ll, table in enumerate(all_tables)]
    else:
        return gto.basis.load(basis, symb="Ne")


# if __name__ == '__main__':
#     mol = gto.Mole()
#     mol.verbose = 5
#     mol.output = None
#     mol.atom = [['He', (0, 0, 0)], ]
#     mol.basis = 'ccpvdz'
#     mol.build(0, 0)

#     def test_model(eigs):
#         assert eigs.shape[-1] == _zeta.size * 9
#         return 1e-3 * torch.sum(eigs, axis=(1,2))
    
#     # SCF Procedure
#     dscf = DSCF(mol, test_model)
#     energy = dscf.kernel()
#     print(energy)
