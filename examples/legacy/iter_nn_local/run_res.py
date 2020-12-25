#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np

# sys.path.append('/path/to/source')
import deepks
from deepks.model.train import main as train_main
from deepks.model.test import main as train_test
from deepks.scf.run import main as scf_main
from deepks.scf.stats import collect_data_grouped
from deepks.utils import load_yaml
from deepks.task.task import PythonTask
from deepks.task.workflow import Sequence, Iteration


niter = 5
nmol = 1500
ntrain = 1000
ntest = 500

train_input = load_yaml('share/train_input.yaml')
scf_input = load_yaml('share/scf_input.yaml')
train_idx = np.arange(ntrain)

task_scf = PythonTask(scf_main, call_kwargs=scf_input,
                      outlog='log.scf',
                      workdir='00.scf',
                      link_prev_files=['model.pth'],
                      share_folder='share', link_share_files=['mol_files.raw'])

task_data = PythonTask(collect_data_grouped, call_args=[train_idx],
                       outlog='log.data',
                       workdir='01.data',
                       link_prev_files=['model.pth', "results"],
                       share_folder='share', link_share_files=['e_ref.npy'])

task_train = PythonTask(train_main, call_args=["old_model.pth"], call_kwargs=train_input,
                        outlog='log.train',
                        workdir='02.train',
                        link_prev_files=[('model.pth', 'old_model.pth'),
                                         'train_paths.raw', 'test_paths.raw'])

seq = Sequence([task_scf, task_data, task_train])
iterate = Iteration(seq, niter, init_folder='share/init', record_file='RECORD')

if os.path.exists('RECORD'):
    iterate.restart()
else:
    iterate.run()
