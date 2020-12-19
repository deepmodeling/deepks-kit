from collections import namedtuple

Field = namedtuple("Field", ["name", "alias", "calc", "shape"])
LabelField = namedtuple("LabelField", ["name", "alias", "calc", "shape", "required_labels"])

def select_fields(names):
    names = [n.lower() for n in names]
    scfs   = [fd for fd in SCF_FIELDS 
                  if fd.name in names 
                  or any(al in names for al in fd.alias)]
    grads  = [fd for fd in GRAD_FIELDS 
                  if fd.name in names 
                  or any(al in names for al in fd.alias)]
    labels = [fd for fd in LABEL_FIELDS
                  if fd.name in names
                  or any(al in names for al in fd.alias)]
    return {"scf": scfs, "grad": grads, "label": labels}


BOHR = 0.52917721092

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
          lambda mf: mf.make_proj_rdms(flatten=True),
          "(nframe, natom, -1)"),
    Field("dm_eig",
          ["eig"],
          lambda mf: mf.make_eig(),
          "(nframe, natom, nproj)"),
    Field("conv", 
          ["converged", "convergence"], 
          lambda mf: mf.converged,
          "(nframe, 1)"),
    Field("mo_coef_occ", 
          ["mo_coeff_occ, orbital_coeff_occ"],
          lambda mf: mf.mo_coeff[:,mf.mo_occ>0].T,
          "(nframe, nao, -1)"),
    Field("mo_ene_occ", 
          ["mo_energy_occ, orbital_ene_occ"],
          lambda mf: mf.mo_energy[mf.mo_occ>0],
          "(nframe, -1)")
]

GRAD_FIELDS = [
    Field("f_base", 
          ["fbase", "force_base", "f0",
           "f_hf", "fhf", "force_hf", 
           "f_ks", "fks", "force_ks"], 
          lambda grad: - grad.get_hf() / BOHR,
          "(nframe, natom, 3)"),
    Field("f_tot", 
          ["f_cf", "fcf", "force_cf", "ftot", "force", "f"], 
          lambda grad: - grad.de / BOHR,
          "(nframe, natom, 3)"),
    Field("gdmx",
          ["grad_dm_x", "grad_pdm_x"],
          lambda grad: grad.make_grad_pdm_x(flatten=True) / BOHR,
          "(nframe, natom, 3, natom, -1)"),
    Field("grad_vx",
          ["grad_eig_x", "geigx", "gvx"],
          lambda grad: grad.make_grad_eig_x() / BOHR,
          "(nframe, natom, 3, natom, nproj)"),
]

LABEL_FIELDS = [
    LabelField("l_e_ref", 
               ["e_ref", "lbl_e_ref", "label_e_ref", "le_ref"],
               lambda res, lbl: lbl["energy"],
               "(nframe, 1)",
               ["energy"]),
    LabelField("l_f_ref", 
               ["f_ref", "lbl_f_ref", "label_f_ref", "lf_ref"],
               lambda res, lbl: lbl["force"],
               "(nframe, natom, 3)",
               ["force"]),
    LabelField("l_e_delta", 
               ["le_delta", "lbl_e_delta", "label_e_delta", "lbl_ed"],
               lambda res, lbl: lbl["energy"] - res["e_base"],
               "(nframe, 1)",
               ["energy"]),
    LabelField("l_f_delta", 
               ["lf_delta", "lbl_f_delta", "label_f_delta", "lbl_fd"],
               lambda res, lbl: lbl["force"] - res["f_base"],
               "(nframe, natom, 3)",
               ["force"]),
    LabelField("err_e", 
               ["e_err", "err_e_tot", "err_e_cf"],
               lambda res, lbl: lbl["energy"] - res["e_tot"],
               "(nframe, 1)",
               ["energy"]),
    LabelField("err_f", 
               ["f_err", "err_f_tot", "err_f_cf"],
               lambda res, lbl: lbl["force"] - res["f_tot"],
               "(nframe, natom, 3)",
               ["force"])
]
