#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import numpy as np

sys.path.append('/home/yixiaoc/SCR/yixiaoc/deep.qc/source_scf')
import deepqc
from deepqc.iter.task import PythonTask
from deepqc.iter.task import ShellTask
from deepqc.iter.task import BatchTask
from deepqc.iter.task import GroupBatchTask
from deepqc.iter.workflow import Sequence
from deepqc.iter.workflow import Iteration

def collect_data(train_idx, test_idx):    
    erefs = np.load('e_ref.npy').reshape(-1)
    systems = sorted(map(os.path.abspath, glob.glob("results/*")))
    assert len(erefs) == len(systems)

    convs = []
    ecfs = []
    for sys_i, ec_i in zip(systems, erefs):
        e0_i = np.load(os.path.join(sys_i, "e_hf.npy"))
        ecc_i = ec_i - e0_i
        np.save(os.path.join(sys_i, "e_cc.npy"), ecc_i)
        convs.append(np.load(os.path.join(sys_i, "conv.npy")))
        ecfs.append(np.load(os.path.join(sys_i, "e_cf.npy")))

    convs = np.array(convs).reshape(-1)
    ecfs = np.array(ecfs).reshape(-1)
    err = erefs - ecfs
    print(f'converged calculation: {np.sum(convs)} / {len(systems)} = {np.sum(convs) / len(systems):.3f}')
    print(f'mean error: {err.mean()}')
    print(f'mean absolute error: {np.abs(err).mean()}')
    print(f'mean absolute error after shift: {np.abs(err - err[train_idx].mean()).mean()}')
    print(f'  training: {np.abs(err[train_idx] - err[train_idx].mean()).mean()}')
    print(f'  testing: {np.abs(err[test_idx] - err[train_idx].mean()).mean()}')

    np.savetxt('train_paths.raw', np.array(systems)[train_idx], fmt='%s')
    np.savetxt('test_paths.raw', np.array(systems)[test_idx], fmt='%s')
    np.savetxt('e_result.out', np.stack([erefs, ecfs], axis=-1), header="real pred")


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
