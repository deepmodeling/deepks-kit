import time
import torch
import numpy as np
from torch import nn
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf, dft
from deepks.scf.scf import t_make_eig, t_make_grad_eig_dm


def t_ele_grad(bfock, c_vir, c_occ, n_occ):
    g = torch.einsum("pa,qi,...pq->...ai", c_vir, c_occ*n_occ, bfock)
    return g.flatten(-2)


def make_grad_eig_egrad(dscf, mo_coeff=None, mo_occ=None, gfock=None):
    if mo_occ is None: 
        mo_occ = dscf.mo_occ
    if mo_coeff is None: 
        mo_coeff = dscf.mo_coeff
    if gfock is None:
        dm = dscf.make_rdm1(mo_coeff, mo_occ)
        if dm.ndim >= 3 and isinstance(dscf, scf.uhf.UHF):
            dm = dm.sum(0)
        gfock = t_make_grad_eig_dm(torch.from_numpy(dm), dscf._t_ovlp_shells).numpy()
    if mo_coeff.ndim >= 3 and mo_occ.ndim >= 2:
        return np.concatenate([make_grad_eig_egrad(dscf, mc, mo, gfock) 
            for mc, mo in zip(mo_coeff, mo_occ)], axis=-1)
    iocc = mo_occ>0
    t_no = torch.from_numpy(mo_occ[iocc]).to(dscf.device)
    t_co = torch.from_numpy(mo_coeff[:, iocc]).to(dscf.device)
    t_cv = torch.from_numpy(mo_coeff[:, ~iocc]).to(dscf.device)
    t_gfock = torch.from_numpy(gfock).to(dscf.device)
    return t_ele_grad(t_gfock, t_cv, t_co, t_no).cpu().numpy()


def gen_coul_loss(dscf, fock=None, ovlp=None, mo_occ=None):
    nao = dscf.mol.nao
    fock = (fock if fock is not None else dscf.get_fock()).reshape(-1, nao, nao)
    s1e = ovlp if ovlp is not None else dscf.get_ovlp()
    mo_occ = (mo_occ if mo_occ is not None else dscf.mo_occ).reshape(-1, nao)
    def _coul_loss_grad(v, target_dm):
        # return coulomb loss and its grad with respect to fock matrix
        # only support single dm, do not use directly for UHF
        a_loss = 0.
        a_grad = 0.
        target_dm = target_dm.reshape(fock.shape)
        for tdm, f1e, nocc in zip(target_dm, fock, mo_occ):
            iocc = nocc>0
            moe, moc = dscf._eigh(f1e+v, s1e)
            eo, ev = moe[iocc], moe[~iocc]
            co, cv = moc[:, iocc], moc[:, ~iocc]
            dm = (co * nocc[iocc]) @ co.T
            # calc loss
            ddm = dm - tdm
            dvj = dscf.get_j(dm=ddm)
            loss = 0.5 * np.einsum("ij,ji", ddm, dvj)
            a_loss += loss
            # calc grad with respect to fock matrix
            ie_mn = 1. / (-ev.reshape(-1, 1) + eo)
            temp_mn = cv.T @ dvj @ co * nocc[iocc] * ie_mn
            dldv = cv @ temp_mn @ co.T
            dldv = dldv + dldv.T
            a_grad += dldv
        return a_loss, a_grad
    return _coul_loss_grad


def make_grad_coul_veig(dscf, target_dm):
    clfn = gen_coul_loss(dscf)
    dm = dscf.make_rdm1()
    if dm.ndim == 3 and isinstance(dscf, scf.uhf.UHF):
        dm = dm.sum(0)
    t_dm = torch.from_numpy(dm).requires_grad_()
    t_eig = t_make_eig(t_dm, dscf._t_ovlp_shells).requires_grad_()
    loss, dldv = clfn(np.zeros_like(dm), target_dm)
    t_veig = torch.zeros_like(t_eig).requires_grad_()
    [t_vc] = torch.autograd.grad(t_eig, t_dm, t_veig, create_graph=True)
    [t_ghead] = torch.autograd.grad(t_vc, t_veig, torch.from_numpy(dldv))
    return t_ghead.detach().cpu().numpy()


def calc_optim_veig(dscf, target_dm, 
                    target_dec=None, gvx=None, 
                    nstep=1, force_factor=1., **optim_args):
    clfn = gen_coul_loss(dscf, fock=dscf.get_fock(vhf=dscf.get_veff0()))
    dm = dscf.make_rdm1()
    if dm.ndim == 3 and isinstance(dscf, scf.uhf.UHF):
        dm = dm.sum(0)
    t_dm = torch.from_numpy(dm).requires_grad_()
    t_eig = t_make_eig(t_dm, dscf._t_ovlp_shells).requires_grad_()
    t_ec = dscf.net(t_eig.to(dscf.device))
    t_veig = torch.autograd.grad(t_ec, t_eig)[0].requires_grad_()
    t_lde = torch.from_numpy(target_dec) if target_dec is not None else None
    t_gvx = torch.from_numpy(gvx) if gvx is not None else None
    # build closure
    def closure():
        [t_vc] = torch.autograd.grad(
            t_eig, t_dm, t_veig, retain_graph=True, create_graph=True)
        loss, dldv = clfn(t_vc.detach().numpy(), target_dm)
        grad = torch.autograd.grad(
            t_vc, t_veig, torch.from_numpy(dldv), only_inputs=True)[0]
        # build closure for force loss
        if t_lde is not None and t_gvx is not None:
            t_pde = torch.tensordot(t_gvx, t_veig)
            lossde = force_factor * torch.sum((t_pde - t_lde)**2)
            grad = grad + torch.autograd.grad(lossde, t_veig, only_inputs=True)[0]
            loss = loss + lossde
        t_veig.grad = grad
        return loss
    # do the optimization
    optim = torch.optim.LBFGS([t_veig], **optim_args)
    tic = (time.clock(), time.time())
    for _ in range(nstep):
        optim.step(closure)
        tic = logger.timer(dscf, 'LBFGS step', *tic)
    logger.note(dscf, f"optimized loss for veig = {closure()}")        
    return t_veig.detach().numpy()


def gcalc_optim_veig(gdscf, target_dm, target_grad, 
                     nstep=1, force_factor=1., **optim_args):
    target_dec = target_grad - gdscf.de0
    gvx = gdscf.make_grad_eig_x()
    return calc_optim_veig(
            gdscf.base,
            target_dm=target_dm, 
            target_dec=target_dec, gvx=gvx, 
            nstep=nstep, force_factor=force_factor, **optim_args)
