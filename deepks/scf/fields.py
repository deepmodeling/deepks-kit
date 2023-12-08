import numpy as np
from typing import List, Callable
from dataclasses import dataclass, field

# Field = namedtuple("Field", ["name", "alias", "calc", "shape"])
# LabelField = namedtuple("LabelField", ["name", "alias", "calc", "shape", "required_labels"])
@dataclass
class Field:
      name: str
      alias: List[str]
      calc: Callable
      shape: str
      required_labels: List[str] = field(default_factory=list)


def select_fields(names):
    names = [n.lower() for n in names]
    scfs   = [fd for fd in SCF_FIELDS 
                  if fd.name in names 
                  or any(al in names for al in fd.alias)]
    grads  = [fd for fd in GRAD_FIELDS 
                  if fd.name in names 
                  or any(al in names for al in fd.alias)]
    return {"scf": scfs, "grad": grads}


def get_field_names(fields):
    names = [fd.name for fd in fields]
    return names


BOHR = 0.52917721092

def isinbohr(mol):
    return mol.unit.upper().startswith(("B", "AU"))

def _Lunit(mol):
    return (1. if isinbohr(mol) else BOHR)

def atom_data(mol):
    raw_data = np.concatenate(
        [mol.atom_charges().reshape(-1,1), mol.atom_coords(unit='Bohr')], 
        axis=1)
    non_ghost = [ii for ii in range(mol.natm) 
        if not mol.elements[ii].startswith("X")]
    return raw_data[non_ghost]


SCF_FIELDS = [
    Field("atom",
          ["atoms", "mol", "molecule"],
          lambda mf: atom_data(mf.mol),
          "(nframe, natom, 4)"),
    Field("e_base", 
          ["ebase", "ene_base", "e0",
           "e_hf", "ehf", "ene_hf", 
           "e_ks", "eks", "ene_ks"], 
          lambda mf: mf.energy_tot0(),
          "(nframe, 1)"),
    Field("e_tot", 
          ["e_cf", "ecf", "ene_cf", "etot", "ene", "energy", "e"],
          lambda mf: mf.e_tot,
          "(nframe, 1)"),
    Field("rdm",
          ["dm"],
          lambda mf: mf.make_rdm1(),
          "(nframe, nao, nao)"),
    Field("proj_dm",
          ["pdm"],
          lambda mf: mf.make_pdm(flatten=True),
          "(nframe, natom, -1)"),
    Field("proj_dm_full",
          ["pdmfull"],
          lambda mf: mf.make_pdm(),
          "(nframe, natom, nproj, nproj)"),
    Field("dm_eig",
          ["eig"],
          lambda mf: mf.make_eig(),
          "(nframe, natom, nproj)"),
    Field("dm_flat",
          ["dmflat"],
          lambda mf: mf.make_flat_pdm(),
          "(nframe, natom, -1)"),
    Field("hcore_eig",
          ["heig"],
          lambda mf: mf.make_eig(mf.get_hcore()),
          "(nframe, natom, nproj)"),
    Field("ovlp_eig",
          ["oeig"],
          lambda mf: mf.make_eig(mf.get_ovlp()),
          "(nframe, natom, nproj)"),
    Field("veff_eig",
          ["veig"],
          lambda mf: mf.make_eig(mf.get_veff()),
          "(nframe, natom, nproj)"),
    Field("fock_eig",
          ["feig"],
          lambda mf: mf.make_eig(mf.get_fock()),
          "(nframe, natom, nproj)"),
    Field("conv", 
          ["converged", "convergence"], 
          lambda mf: mf.converged,
          "(nframe, 1)"),
    Field("mo_coef_occ", # do not support UHF
          ["mo_coeff_occ, orbital_coeff_occ"],
          lambda mf: mf.mo_coeff[:,mf.mo_occ>0].T,
          "(nframe, -1, nao)"),
    Field("mo_ene_occ", # do not support UHF
          ["mo_energy_occ, orbital_ene_occ"],
          lambda mf: mf.mo_energy[mf.mo_occ>0],
          "(nframe, -1)"),
    Field("o_base", # do not support UHF
          ["mo_energy_occ, orbital_ene_occ"],
          lambda mf: mf.mo_energy0()[np.argmax(mf.mo_energy[mf.mo_occ>0])+1] - mf.mo_energy0()[np.argmax(mf.mo_energy[mf.mo_occ>0])],
          "(nframe, -1)"),
    Field("o_tot", # do not support UHF
          ["orbital, mo_energy_occ, orbital_ene_occ"],
          lambda mf: mf.mo_energy()[np.argmax(mf.mo_energy[mf.mo_occ>0])+1] - mf.mo_energy()[np.argmax(mf.mo_energy[mf.mo_occ>0])],
          "(nframe, -1)"),
    Field("orbital_precalc", # do not support UHF
          ["orbital_coef_precalc"],
          lambda mf: mf.make_orbital_precalc(),
          "(nframe, -1, natom, nproj)"),
    # below are fields that requires labels
    Field("l_e_ref", 
          ["e_ref", "lbl_e_ref", "label_e_ref", "le_ref"],
          lambda mf, **lbl: lbl["energy"],
          "(nframe, 1)",
          ["energy"]),
    Field("l_e_delta", 
          ["le_delta", "lbl_e_delta", "label_e_delta", "lbl_ed"],
          lambda mf, **lbl: lbl["energy"] - mf.energy_tot0(),
          "(nframe, 1)",
          ["energy"]),
    Field("err_e", 
          ["e_err", "err_e_tot", "err_e_cf"],
          lambda mf, **lbl: lbl["energy"] - mf.e_tot,
          "(nframe, 1)",
          ["energy"]),
    Field("l_o_delta",
          ["lo_delta", "lbl_o_delta", "label_o_delta", "lbl_od"],
          lambda mf, **lbl: lbl["orbital"] -  (mf.mo_energy0()[np.argmax(mf.mo_energy[mf.mo_occ>0])+1] - mf.mo_energy0()[np.argmax(mf.mo_energy[mf.mo_occ>0])]), #last two terms gives current band gap, i.e. LUMO - HOMO
          "(nframe, -1)",
          ["orbital"]),
]

GRAD_FIELDS = [
    Field("f_base", 
          ["fbase", "force_base", "f0",
           "f_hf", "fhf", "force_hf", 
           "f_ks", "fks", "force_ks"], 
          lambda grad: - grad.get_base() / _Lunit(grad.mol),
          "(nframe, natom_raw, 3)"),
    Field("f_tot", 
          ["f_cf", "fcf", "force_cf", "ftot", "force", "f"], 
          lambda grad: - grad.de / _Lunit(grad.mol),
          "(nframe, natom_raw, 3)"),
    Field("grad_dmx",
          ["grad_dm_x", "grad_pdm_x"],
          lambda grad: grad.make_grad_pdm_x(flatten=True) / _Lunit(grad.mol),
          "(nframe, natom_raw, 3, natom, -1)"),
    Field("grad_vx",
          ["grad_eig_x", "geigx", "gvx"],
          lambda grad: grad.make_grad_eig_x()  / _Lunit(grad.mol),
          "(nframe, natom_raw, 3, natom, nproj)"),
    # below are fields that requires labels
    Field("l_f_ref", 
          ["f_ref", "lbl_f_ref", "label_f_ref", "lf_ref"],
          lambda grad, **lbl: lbl["force"],
          "(nframe, natom_raw, 3)",
          ["force"]),
    Field("l_f_delta", 
          ["lf_delta", "lbl_f_delta", "label_f_delta", "lbl_fd"],
          lambda grad, **lbl: lbl["force"] - (-grad.get_base() / _Lunit(grad.mol)),
          "(nframe, natom_raw, 3)",
          ["force"]),
    Field("err_f", 
          ["f_err", "err_f_tot", "err_f_cf"],
          lambda grad, **lbl: lbl["force"] - (-grad.de / _Lunit(grad.mol)),
          "(nframe, natom_raw, 3)",
          ["force"]),
]


# below are additional methods from addons
from deepks.scf import addons

SCF_FIELDS.extend([
    # the following two are used for regularizing the potential
    Field("grad_veg",
          ["grad_eig_egrad", "jac_eig_egrad"],
          lambda mf: addons.make_grad_eig_egrad(mf),
          "(nframe, natom, nproj, -1)"),
    Field("eg_base",
          ["ele_grad_base", "egrad0", "egrad_base"],
          lambda mf: mf.get_grad0(),
          "(nframe, -1)"),
    # the following one is used for coulomb loss optimization
    Field("grad_ldv",
          ["grad_coul_dv", "grad_coul_deig", "coulomb_grad"], 
          lambda mf, **lbl: addons.make_grad_coul_veig(mf, target_dm=lbl["dm"]),
          "(nframe, natom, nproj)",
          ["dm"]),
    Field("l_veig_raw",
          ["optim_veig_raw", "l_opt_v_raw", "l_optim_veig_raw"], 
          lambda mf, **lbl: addons.calc_optim_veig(mf, lbl["dm"], nstep=1),
          "(nframe, natom, nproj)",
          ["dm"]),
])

GRAD_FIELDS.extend([
    # the following one is used for coulomb loss optimization from grad class
    Field("l_veig",
          ["optim_veig", "l_opt_v", "l_optim_veig"], 
          lambda grad, **lbl: addons.gcalc_optim_veig(
              grad, lbl["dm"], -_Lunit(grad.mol)*lbl["force"], nstep=1),
          "(nframe, natom, nproj)",
          ["dm", "force"]),
    Field("l_veig_nof",
          ["optim_veig_nof", "l_opt_v_nof", "l_optim_veig_nof"], 
          lambda grad, **lbl: addons.gcalc_optim_veig(
              grad, lbl["dm"], grad.de, nstep=1),
          "(nframe, natom, nproj)",
          ["dm"]),
])
