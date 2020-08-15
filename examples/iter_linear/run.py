import os
import sys
import torch
import numpy as np
import pyscf
from pyscf import gto
from sklearn import linear_model

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')
from deepqc.scf.scf import DeepSCF
from deepqc.scf.main import build_mol, solve_mol

def get_linear_model(weig, wec):
#     too_small = weig.reshape(-1,108).std(0) < 1e-3
    wreg = linear_model.Ridge(1e-7, tol=1e-9)
    wreg.fit(weig.sum(1)[:], wec[:])
    linear = torch.nn.Linear(108,1).double()
    linear.weight.data[:] = torch.from_numpy(wreg.coef_)
    linear.bias.data[:] = torch.tensor(wreg.intercept_ / 3)
    model = lambda x: linear(x).sum(1)
    return model

def get_linear_model_normed(weig, wec, stdmin=1e-3):
#     too_small = weig.reshape(-1,108).std(0) < 1e-3
    input_scale = weig.reshape(-1,108).std(0).clip(stdmin)
    t_input_scale = torch.from_numpy(input_scale)
    weig /= input_scale
    wreg = linear_model.Ridge(1e-7, tol=1e-9)
    wreg.fit(weig.sum(1)[:], wec[:])
    linear = torch.nn.Linear(108,1).double()
    linear.weight.data[:] = torch.from_numpy(wreg.coef_)
    linear.bias.data[:] = torch.tensor(wreg.intercept_ / 3)
    model = lambda x: linear(x / t_input_scale).sum(1)
    return model

nmol = 1000
ntrain = 900
niter = 10

mol_list = [build_mol(f'../../../data/tom_miller/water/geometry/{i:0>5}.xyz') for i in range(nmol)]
ehfs = np.load('../../../data/tom_miller/water/rproj_mb2/e_hf.npy').reshape(-1)[:nmol]
wene = np.loadtxt('../../../data/tom_miller/water/energy.dat', usecols=(1,2,3,4))[:nmol]
erefs = wene[:,3]
ecfs = ehfs
ecs = erefs - ehfs
ceigs = np.load('../../../data/tom_miller/water/rproj_mb2/dm_eig.npy')[:nmol]
model = get_linear_model(ceigs[:ntrain], ecs[:ntrain])

os.makedirs('dump', exist_ok=True)
np.save('dump/000.ehfs.npy', ehfs)
np.save('dump/000.ecfs.npy', ecfs)
np.save('dump/000.ceigs.npy', ceigs)
np.save('dump/000.ecs.npy', ecs)
np.save('dump/000.convs.npy', np.ones(ehfs.shape, dtype=bool))

for i in range(1, niter+1):
    oldecfs, oldceigs, oldehfs = ecfs, ceigs, ehfs
    oldecs = ecs
    oldmodel = model
    
    results = [solve_mol(mol, model) for mol in mol_list]
    meta, ehfs, ecfs, cdms, ceigs, convs = map(np.array, zip(*results))
    ecs = erefs - ehfs
    model = get_linear_model(ceigs[:ntrain], ecs[:ntrain])
    
    print((ecfs - erefs).mean(), np.abs(ecfs - erefs).mean())
    
    np.save(f'dump/{i:0>3}.ehfs.npy', ehfs)
    np.save(f'dump/{i:0>3}.ecfs.npy', ecfs)
    np.save(f'dump/{i:0>3}.ceigs.npy', ceigs)
    np.save(f'dump/{i:0>3}.ecs.npy', ecs)
    np.save(f'dump/{i:0>3}.convs.npy', convs)
