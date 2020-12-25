#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np

# sys.path.append('/path/to/source')
import deepqc
from deepqc.model.train import main as train_main
from deepqc.scf.run import main as scf_main
from deepqc.utils import load_yaml
from deepqc.task.task import PythonTask
from deepqc.task.workflow import Sequence, Iteration

from pathlib import Path
import shutil

def collect_data(nmol, ntrain):
    ecf = np.load('results/e_tot.npy')
    assert ecf.size == nmol
    eref = np.load('e_ref.npy')

    err = eref.reshape(-1) - ecf.reshape(-1)
    convs = np.load("results/conv.npy").reshape(-1)
    print(f'converged calculation: {np.sum(convs)} / {nmol} = {np.sum(convs) / nmol:.3f}')
    print(f'mean error: {err.mean()}')
    print(f'mean absolute error: {np.abs(err).mean()}')
    print(f'mean absolute error after shift: {np.abs(err - err[:ntrain].mean()).mean()}')
    print(f'  training: {np.abs(err[:ntrain] - err[:ntrain].mean()).mean()}')
    print(f'  testing: {np.abs(err[ntrain:] - err[:ntrain].mean()).mean()}')

    ehf = np.load('results/e_base.npy')
    np.save('results/l_e_delta.npy', eref - ehf)

    dd = ['dm_eig.npy', 'l_e_delta.npy']
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    for d in dd:
        np.save(f"train/{d}", np.load(f'results/{d}')[:ntrain])
    for d in dd:
        np.save(f"test/{d}", np.load(f'results/{d}')[ntrain:])
    shutil.copy('results/system.raw', 'train')
    shutil.copy('results/system.raw', 'test')
    Path('train_paths.raw').write_text(str(Path('train').absolute()))
    Path('test_paths.raw').write_text(str(Path('test').absolute()))


niter = 5
nmol = 1500
ntrain = 1000
ntest = 500

train_input = load_yaml('share/train_input.yaml')
scf_input = load_yaml('share/scf_input.yaml')

task_train = PythonTask(train_main, call_kwargs=train_input,
                        outlog='log.train',
                        workdir='00.train',
                        link_prev_files=['train_paths.raw', 'test_paths.raw'])

task_scf = PythonTask(scf_main, call_kwargs=scf_input,
                      outlog='log.scf',
                      workdir='01.scf',
                      link_prev_files=['model.pth'],
                      share_folder='share', link_share_files=['mol_files.raw'])

task_data = PythonTask(collect_data, call_args=[nmol, ntrain],
                       outlog='log.data',
                       workdir='02.data',
                       link_prev_files=['results'],
                       share_folder='share', link_share_files=['e_ref.npy'])

seq = Sequence([task_train, task_scf, task_data])
iterate = Iteration(seq, niter, init_folder='share/init', record_file='RECORD')

if Path('RECORD').exists():
    iterate.restart()
else:
    iterate.run()
