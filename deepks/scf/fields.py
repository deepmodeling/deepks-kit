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


BOHR = 0.52917721092

def isinbohr(mol):
    return mol.unit.upper().startswith(("B", "AU"))


SCF_FIELDS = [
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
    Field("dm_eig",
          ["eig"],
          lambda mf: mf.make_eig(),
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
    # the following one is used for coulomb loss optimization
    Field("l_veig",
          ["optim_veig", "l_opt_v", "l_optim_veig"], 
          lambda mf, **lbl: mf.calc_optim_veig(lbl["dm"], nstep=2),
          "(nframe, natom, nproj)",
          ["dm"]),
]

GRAD_FIELDS = [
    Field("f_base", 
          ["fbase", "force_base", "f0",
           "f_hf", "fhf", "force_hf", 
           "f_ks", "fks", "force_ks"], 
          lambda grad: - grad.get_base() 
                       / (1. if isinbohr(grad.mol) else BOHR),
          "(nframe, natom, 3)"),
    Field("f_tot", 
          ["f_cf", "fcf", "force_cf", "ftot", "force", "f"], 
          lambda grad: - grad.de 
                       / (1. if isinbohr(grad.mol) else BOHR),
          "(nframe, natom, 3)"),
    Field("grad_dmx",
          ["grad_dm_x", "grad_pdm_x"],
          lambda grad: grad.make_grad_pdm_x(flatten=True) 
                       / (1. if isinbohr(grad.mol) else BOHR),
          "(nframe, natom, 3, natom, -1)"),
    Field("grad_vx",
          ["grad_eig_x", "geigx", "gvx"],
          lambda grad: grad.make_grad_eig_x()  
                       / (1. if isinbohr(grad.mol) else BOHR),
          "(nframe, natom, 3, natom, nproj)"),
    # below are fields that requires labels
    Field("l_f_ref", 
          ["f_ref", "lbl_f_ref", "label_f_ref", "lf_ref"],
          lambda grad, **lbl: lbl["force"],
          "(nframe, natom, 3)",
          ["force"]),
    Field("l_f_delta", 
          ["lf_delta", "lbl_f_delta", "label_f_delta", "lbl_fd"],
          lambda grad, **lbl: lbl["force"] 
              - (-grad.get_base() / (1. if isinbohr(grad.mol) else BOHR)),
          "(nframe, natom, 3)",
          ["force"]),
    Field("err_f", 
          ["f_err", "err_f_tot", "err_f_cf"],
          lambda grad, **lbl: lbl["force"] 
              - (-grad.de / (1. if isinbohr(grad.mol) else BOHR)),
          "(nframe, natom, 3)",
          ["force"]),
]
