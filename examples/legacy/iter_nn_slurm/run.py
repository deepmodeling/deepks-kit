#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import numpy as np

sys.path.append('/home/yixiaoc/SCR/yixiaoc/deep.qc/source_scf')
import deepqc
from deepqc.task.task import PythonTask
from deepqc.task.task import ShellTask
from deepqc.task.task import BatchTask
from deepqc.task.task import GroupBatchTask
from deepqc.task.workflow import Sequence
from deepqc.task.workflow import Iteration
from deepqc.scf.stats import collect_data


niter = 20

# Define Training
nmodel = 4

train_res = {"time_limit": "24:00:00",
             "mem_limit": 32,
             "numb_gpu": 1}

train_cmd = "python -u ~/SCR/yixiaoc/deep.qc/source_scf/deepqc/train/main.py input.yaml"

batch_train = [BatchTask(cmds=train_cmd, 
                         workdir=f'task.{i:02}',
                         share_folder="share",
                         link_share_files=["input.yaml"], 
                         link_prev_files=['train_paths.raw', 'test_paths.raw'])
               for i in range(nmodel)]
run_train = GroupBatchTask(batch_train, 
                           resources=train_res, 
                           outlog="log.train")

post_train = ShellTask("ln -s task.00/model.pth .")

clean_train = ShellTask("rm slurm-*.out")

train_flow = Sequence([run_train, post_train, clean_train], workdir='00.train')


# Define SCF
ngroup = 24
ntrain = 3000

mol_files = np.loadtxt('share/mol_files.raw', dtype=str)
group_files = [mol_files[i::ngroup] for i in range(ngroup)]

envs = {"PYSCF_MAX_MEMORY": 32000}
scf_res = {"cpus_per_task": 5,
           "time_limit": "24:00:00",
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
    "python -u ~/SCR/yixiaoc/deep.qc/source_scf/deepqc/scf/main.py",
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
