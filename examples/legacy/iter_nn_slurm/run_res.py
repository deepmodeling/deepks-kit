#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import numpy as np

# sys.path.append('/path/to/source')
import deepks
from deepks.task.task import PythonTask
from deepks.task.task import ShellTask
from deepks.task.task import BatchTask
from deepks.task.task import GroupBatchTask
from deepks.task.workflow import Sequence
from deepks.task.workflow import Iteration
from deepks.scf.stats import collect_data


niter = 5
ntrain = 7000

# Define Training
nmodel = 4

train_res = {"time_limit": "6:00:00",
             "mem_limit": 32,
             "numb_gpu": 1}

train_cmd = "python -u /path/to/source/deepks/train/main.py input.yaml --restart ../old_model.pth"

batch_train = [BatchTask(cmds=train_cmd, 
                         workdir=f'task.{i:02}',
                         share_folder="share",
                         link_share_files=["input.yaml"], 
                         link_prev_files=['train_paths.raw', 'test_paths.raw'])
               for i in range(nmodel)]
run_train = GroupBatchTask(batch_train, 
                           resources=train_res, 
                           outlog="log.train",
                           link_prev_files=[('model.pth', 'old_model.pth')])

post_train = ShellTask("ln -s task.00/model.pth .")

clean_train = ShellTask("rm slurm-*.out")

train_flow = Sequence([run_train, post_train, clean_train], workdir='00.train')


# Define SCF
ngroup = 12

mol_files = np.loadtxt('share/mol_files.raw', dtype=str)
group_files = [mol_files[i::ngroup] for i in range(ngroup)]

envs = {"PYSCF_MAX_MEMORY": 32000}
scf_res = {"cpus_per_task": 5,
           "time_limit": "6:00:00",
           "mem_limit": 32,
           "envs": envs}

remote = {"work_path": '/home/yixiaoc/SCR/yixiaoc/tmp',
          "hostname": "della",
          "username": "yixiaoc",
          "port": 22}
disp = {"context_type": 'ssh',
        "batch_type": 'slurm',
        "remote_profile": remote}

cmd_templ = " ".join([
    "python -u /path/to/source/deepks/scf/main.py",
    "{mol_files}",
    "-m ../model.pth",
    "-d ../results",
    "-B ccpvdz",
    "--verbose 1",
    "--conv-tol 1e-6", 
    "--conv-tol-grad 3e-2"
])

batch_scf = [BatchTask(cmds=cmd_templ.format(mol_files=" ".join(gf)),
                       workdir=f'task.{i:02}',
                       backward_files=['log.scf', 'err'])
             for i, gf in enumerate(group_files)]
run_scf = GroupBatchTask(batch_scf, 
                         dispatcher=disp,
                         resources=scf_res, 
                         outlog="log.scf",
                         link_prev_files=['model.pth'],
                         forward_files=['model.pth'],
                         backward_files=['results/*'])

all_idx = np.loadtxt('share/index.raw', dtype=int)
train_idx = all_idx[:ntrain]
test_idx = all_idx[ntrain:]

post_scf = PythonTask(collect_data, call_args=[train_idx, test_idx],
                      call_kwargs={"sys_dir": "results", "ene_ref": "e_ref.npy"},
                      outlog='log.data',
                      share_folder='share', 
                      link_share_files=['e_ref.npy'])

clean_scf = ShellTask("rm slurm-*.out")

scf_flow = Sequence([run_scf, post_scf, clean_scf], workdir='01.scf')


# Group them together
per_iter = Sequence([train_flow, scf_flow])
iterate = Iteration(per_iter, niter, init_folder='share/init', record_file='RECORD')

if os.path.exists('RECORD'):
    iterate.restart()
else:
    iterate.run()
