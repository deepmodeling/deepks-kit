import os
import sys
import numpy as np
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
try:
    import deepks
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepks.model.model import CorrNet
from deepks.model.reader import GroupReader
from deepks.utils import load_dirs


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess(model, g_reader, 
                preshift=True, prescale=False, prescale_sqrt=False, prescale_clip=0,
                prefit=True, prefit_ridge=10, prefit_trainable=False):
    shift = model.input_shift.cpu().detach().numpy()
    scale = model.input_scale.cpu().detach().numpy()
    symm_sec = model.shell_sec # will be None if no embedding
    prefit_trainable = prefit_trainable and symm_sec is None # no embedding
    if preshift or prescale:
        davg, dstd = g_reader.compute_data_stat(symm_sec)
        if preshift: 
            shift = davg
        if prescale: 
            scale = dstd
            if prescale_sqrt: 
                scale = np.sqrt(scale)
            if prescale_clip: 
                scale = scale.clip(prescale_clip)
        model.set_normalization(shift, scale)
    if prefit:
        weight, bias = g_reader.compute_prefitting(
            shift=shift, scale=scale, 
            ridge_alpha=prefit_ridge, symm_sections=symm_sec)
        model.set_prefitting(weight, bias, trainable=prefit_trainable)


def make_loss(cap=None, shrink=None, reduction="mean"):
    def loss_fn(input, target):
        diff = target - input
        if shrink and shrink > 0:
            diff = F.softshrink(diff, shrink)
        sqdf = diff ** 2
        if cap and cap > 0:
            abdf = diff.abs()
            sqdf = torch.where(abdf < cap, sqdf, cap * (2*abdf - cap))
        if reduction is None or reduction.lower() == "none":
            return sqdf
        elif reduction.lower() == "mean":
            return sqdf.mean()
        elif reduction.lower() == "sum":
            return sqdf.sum()
        elif reduction.lower() in ("batch", "bmean"):
            return sqdf.sum() / sqdf.shape[0]
        else:
            raise ValueError(f"{reduction} is not a valid reduction type")
    return loss_fn

# equiv to nn.MSELoss()
L2LOSS = make_loss(cap=None, shrink=None, reduction="mean")


class Evaluator:
    def __init__(self,
                 energy_factor=1., force_factor=0., 
                 stress_factor=0., orbital_factor=0.,
                 density_factor=0., grad_penalty=0., 
                 energy_lossfn=None, force_lossfn=None, 
                 stress_lossfn=None, orbital_lossfn=None):
        # energy term
        if energy_lossfn is None:
            energy_lossfn = {}
        if isinstance(energy_lossfn, dict):
            energy_lossfn = make_loss(**energy_lossfn)
        self.e_factor = energy_factor
        self.e_lossfn = energy_lossfn
        # force term
        if force_lossfn is None:
            force_lossfn = {}
        if isinstance(force_lossfn, dict):
            force_lossfn = make_loss(**force_lossfn)
        self.f_factor = force_factor
        self.f_lossfn = force_lossfn
        # stress term
        if stress_lossfn is None:
            stress_lossfn = {}
        if isinstance(stress_lossfn, dict):
            stress_lossfn = make_loss(**stress_lossfn)
        self.s_factor = stress_factor
        self.s_lossfn = stress_lossfn
         # orbital(bandgap) term
        if orbital_lossfn is None:
            orbital_lossfn = {}
        if isinstance(orbital_lossfn, dict):
            orbital_lossfn = make_loss(**orbital_lossfn)
        self.o_factor = orbital_factor
        self.o_lossfn = orbital_lossfn
        # coulomb term of dm; requires head gradient
        self.d_factor = density_factor
        # gradient penalty, not very useful
        self.g_penalty = grad_penalty

    def __call__(self, model, sample):
        _dref = next(model.parameters())
        tot_loss = 0.
        sample = {k: v.to(_dref, non_blocking=True) for k, v in sample.items()}
        e_label, eig = sample["lb_e"], sample["eig"]
        nframe = e_label.shape[0]
        requires_grad =  ( (self.f_factor > 0 and "lb_f" in sample) 
                        or (self.s_factor > 0 and "lb_s" in sample) 
                        or (self.o_factor > 0 and "lb_o" in sample)
                        or (self.d_factor > 0 and "gldv" in sample)
                        or self.g_penalty > 0)
        eig.requires_grad_(requires_grad)
        # begin the calculation
        e_pred = model(eig)
        tot_loss = tot_loss + self.e_factor * self.e_lossfn(e_pred, e_label)
        if requires_grad:
            [gev] = torch.autograd.grad(e_pred, eig, 
                        grad_outputs=torch.ones_like(e_pred),
                        retain_graph=True, create_graph=True, only_inputs=True)
            # for now always use pure l2 loss for gradient penalty
            if self.g_penalty > 0 and "eg0" in sample:
                eg_base, gveg = sample["eg0"], sample["gveg"]
                eg_tot = torch.einsum('...apg,...ap->...g', gveg, gev) + eg_base
                tot_loss = tot_loss + self.g_penalty * eg_tot.pow(2).mean(0).sum()
            # optional force calculation
            if self.f_factor > 0 and "lb_f" in sample:
                f_label, gvx = sample["lb_f"], sample["gvx"]
                f_pred = - torch.einsum("...bxap,...ap->...bx", gvx, gev)
                tot_loss = tot_loss + self.f_factor * self.f_lossfn(f_pred, f_label)
            # optional stress calculation
            if self.s_factor > 0 and "lb_s" in sample:
                s_label, gvepsl = sample["lb_s"], sample["gvepsl"]
                s_pred = torch.einsum("...iap,...ap->...i", gvepsl, gev)
                tot_loss = tot_loss + self.s_factor * self.s_lossfn(s_pred, s_label)
            # optional orbital(bandgap) calculation
            if self.o_factor > 0 and "lb_o" in sample:
                o_label, op = sample["lb_o"], sample["op"]
                o_pred = torch.einsum("...iap,...ap->...i", op, gev)
                tot_loss = tot_loss + self.o_factor * self.o_lossfn(o_pred, o_label)
            # density loss with fix head grad
            if self.d_factor > 0 and "gldv" in sample:
                gldv = sample["gldv"]
                tot_loss = tot_loss + self.d_factor * (gldv * gev).mean(0).sum()
        return tot_loss


def train(model, g_reader, n_epoch=1000, test_reader=None, *,
          energy_factor=1., force_factor=0., stress_factor=0., orbital_factor=0., density_factor=0.,
          energy_loss=None, force_loss=None, stress_loss=None, orbital_loss=None, grad_penalty=0.,
          start_lr=0.001, decay_steps=100, decay_rate=0.96, stop_lr=None,
          weight_decay=0.,  fix_embedding=False,
          display_epoch=100, ckpt_file="model.pth",
          graph_file=None, device=DEVICE):
    
    model = model.to(device)
    model.eval()
    print("# working on device:", device)
    if test_reader is None:
        test_reader = g_reader
    # fix parameters if needed
    if fix_embedding and model.embedder is not None:
        model.embedder.requires_grad_(False)
    # set up optimizer and lr scheduler
    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)
    if stop_lr is not None:
        decay_rate = (stop_lr / start_lr) ** (1 / (n_epoch // decay_steps))
        print(f"# resetting decay_rate: {decay_rate:.4f} "
              + f"to satisfy stop_lr: {stop_lr:.2e}")
    scheduler = optim.lr_scheduler.StepLR(optimizer, decay_steps, decay_rate)
    # make evaluators for training
    evaluator = Evaluator(energy_factor=energy_factor, force_factor=force_factor, 
                          stress_factor=stress_factor, orbital_factor=orbital_factor,
                          energy_lossfn=energy_loss, force_lossfn=force_loss,
                          stress_lossfn=stress_loss, orbital_lossfn=orbital_loss,
                          density_factor=density_factor, grad_penalty=grad_penalty)
    # make test evaluator that only returns l2loss of energy
    test_eval = Evaluator(energy_factor=1., energy_lossfn=L2LOSS, 
                          force_factor=0., density_factor=0., grad_penalty=0.)

    print("# epoch      trn_err   tst_err        lr  trn_time  tst_time ")
    tic = time()
    trn_loss = np.mean([evaluator(model, batch).item() 
                    for batch in g_reader.sample_all_batch()])
    tst_loss = np.mean([test_eval(model, batch).item() 
                    for batch in test_reader.sample_all_batch()])
    tst_time = time() - tic
    print(f"  {0:<8d}  {np.sqrt(np.abs(trn_loss)):>.2e}  {np.sqrt(np.abs(tst_loss)):>.2e}"
          f"  {start_lr:>.2e}  {0:>8.2f}  {tst_time:>8.2f}")

    for epoch in range(1, n_epoch+1):
        tic = time()
        loss_list = []
        for sample in g_reader:
            model.train()
            optimizer.zero_grad()
            loss = evaluator(model, sample)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        scheduler.step()

        if epoch % display_epoch == 0:
            model.eval()
            trn_loss = np.mean(loss_list)
            trn_time = time() - tic
            tic = time()
            tst_loss = np.mean([test_eval(model, batch).item() 
                            for batch in test_reader.sample_all_batch()])
            tst_time = time() - tic
            print(f"  {epoch:<8d}  {np.sqrt(np.abs(trn_loss)):>.2e}  {np.sqrt(np.abs(tst_loss)):>.2e}"
                  f"  {scheduler.get_last_lr()[0]:>.2e}  {trn_time:>8.2f}  {tst_time:8.2f}")
            if ckpt_file:
                model.save(ckpt_file)

    if graph_file:
        model.compile_save(graph_file)
    

def main(train_paths, test_paths=None,
         restart=None, ckpt_file=None, 
         model_args=None, data_args=None, 
         preprocess_args=None, train_args=None, 
         proj_basis=None, seed=None, device=None):
   
    if seed is None: 
        seed = np.random.randint(0, 2**32)
    print(f'# using seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model_args is None: model_args = {}
    if data_args is None: data_args = {}
    if preprocess_args is None: preprocess_args = {}
    if train_args is None: train_args = {}
    if proj_basis is not None:
        model_args["proj_basis"] = proj_basis
    if ckpt_file is not None:
        train_args["ckpt_file"] = ckpt_file
    if device is not None:
        train_args["device"] = device

    train_paths = load_dirs(train_paths)
    # print(f'# training with {len(train_paths)} system(s)')
    g_reader = GroupReader(train_paths, **data_args)
    if test_paths is not None:
        test_paths = load_dirs(test_paths)
        # print(f'# testing with {len(test_paths)} system(s)')
        test_reader = GroupReader(test_paths, **data_args)
    else:
        print('# testing with training set')
        test_reader = None

    if restart is not None:
        model = CorrNet.load(restart)
    else:
        input_dim = g_reader.ndesc
        if model_args.get("input_dim", input_dim) != input_dim:
            print(f"# `input_dim` in `model_args` does not match data",
                  "({input_dim}).", "Use the one in data.", file=sys.stderr)
        model_args["input_dim"] = input_dim
        model = CorrNet(**model_args).double()
        
    preprocess(model, g_reader, **preprocess_args)
    train(model, g_reader, test_reader=test_reader, **train_args)


if __name__ == "__main__":
    from deepks.main import train_cli as cli
    cli()