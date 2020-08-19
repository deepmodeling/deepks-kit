#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np

sys.path.append('/home/yixiaoc/SCR/yixiaoc/deep.qc/source_scf')
import deepqc
from deepqc.scf.tools import collect_data_grouped
from deepqc.task.task import PythonTask, BatchTask, GroupBatchTask
from deepqc.task.workflow import Sequence, Iteration

nsys = 1
niter = 25
ntrain = 1000
train_idx = np.arange(ntrain)

# SCF

scf_cmd_tmpl = " ".join([
    "python -u ~/SCR/yixiaoc/deep.qc/source_scf/deepqc/scf/main.py",
    "scf_input.yaml",
    "-m model.pth",
    "-s mol_files.raw",
    "-d results"])

envs = {"PYSCF_MAX_MEMORY": 16000}
scf_res = {"cpus_per_task": 10,
           "time_limit": "6:00:00",
           "mem_limit": 16,
           "envs": envs}

task_scf = GroupBatchTask(
                [BatchTask(scf_cmd_tmpl.format(i=i),
                           workdir=".", #f'task.{i}',
                           share_folder='share', 
                           link_share_files=['mol_files.raw', 
                                             ('raw_scf_input.yaml', 'scf_input.yaml')])
                    for i in range(nsys)],
                workdir='00.scf',
                outlog='log.scf',
                resources=scf_res,
                link_prev_files=['model.pth'])

# labeling

task_data = PythonTask(
                lambda: [collect_data_grouped(train_idx=train_idx,
                                              append=True,
                                              ene_ref=f"e_ref.npy",
                                              force_ref=f"f_ref.npy",
                                              sys_dir=f"results") 
                         for i in range(nsys)],
                outlog='log.data',
                workdir='01.data',
                link_prev_files=['model.pth'] + [f"results" for i in range(nsys)],
                share_folder='share', 
                link_share_files=[f'e_ref.npy' for i in range(nsys)]
                                +[f'f_ref.npy' for i in range(nsys)])

# training

train_cmd = " ".join([
    "python -u ~/SCR/yixiaoc/deep.qc/source_scf/deepqc/train/main.py",
    "train_input.yaml",
    "--restart old_model.pth"])

train_res = {"time_limit": "24:00:00",
             "mem_limit": 32,
             "numb_gpu": 1}

task_train = BatchTask(cmds=train_cmd,
                       outlog='log.train',
                       workdir='02.train',
                       resources=train_res, 
                       link_prev_files=[('model.pth', 'old_model.pth'),
                                        'train_paths.raw', 'test_paths.raw'],
                       share_folder = 'share',
                       link_share_files=["train_input.yaml"])

# combine

seq = Sequence([task_scf, task_data, task_train])
iterate = Iteration(seq, niter, init_folder='share/init', record_file='RECORD')

if os.path.exists('RECORD'):
    iterate.restart()
else:
    iterate.run()
