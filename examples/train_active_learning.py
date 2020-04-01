#!/usr/bin/env python
# coding: utf-8

import os
import sys
from glob import glob
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import deepqc
from deepqc.iter.task import PythonTask
from deepqc.iter.task import ShellTask
from deepqc.iter.task import BatchTask
from deepqc.iter.task import GroupBatchTask
from deepqc.iter.workflow import Sequence
from deepqc.iter.workflow import Iteration


# define key parameters
nsel = 200
nmodel = 4
niter = 21

# define training task
train_res = {"time_limit": "24:00:00",
             "mem_limit": 32,
             "numb_gpu": 1}

disp = {"context_type": 'local',
        "batch_type": 'slurm'}

train_cmd = "python -u ~/SCR/yixiaoc/deep.qc/source_scf/deepqc/train/main.py input.yaml"

batch_train = [BatchTask(cmds=train_cmd, 
                         workdir=f'model.{i:02}',
                         share_folder="share",
                         link_share_files=["input.yaml"])
               for i in range(nmodel)]
task_train = GroupBatchTask(batch_train, 
                           resources=train_res,
                           dispatcher=disp,
                           outlog="log.train",
                           errlog="err.train",
                           link_prev_files=[('new_train_paths.raw', 'train_paths.raw'),
                                            ('new_test_paths.raw', 'test_paths.raw')])


# define testing task
test_cmd = "srun -N 1 -t 1:00:00 --gres=gpu:1 bash test_model.sh 1> log.test 2> err.test"
task_test = ShellTask(test_cmd,
                      share_folder="share",
                      link_share_files=["test_model.sh"])


#define selecting task
def select_data(nsel):
    paths = glob("model.*")
    old_trn = np.loadtxt("train_paths.raw", dtype=str)
    old_tst = np.loadtxt("test_paths.raw", dtype=str)
    trn_res = np.stack([np.loadtxt(f"{m}/test/train.all.out")[:,1] for m in paths], -1)
    tst_res = np.stack([np.loadtxt(f"{m}/test/test.all.out")[:,1] for m in paths], -1)

    tst_std = np.std(tst_res, axis=-1)
    order = np.argsort(tst_std)[::-1]
    sel = order[:nsel]
    rst = np.sort(order[nsel:])

    new_trn = np.concatenate([old_trn, old_tst[sel]])
    new_tst = old_tst[rst]
    np.savetxt("new_train_paths.raw", new_trn, fmt="%s")
    np.savetxt("new_test_paths.raw", new_tst, fmt="%s")
    
task_select = PythonTask(select_data, call_args=[nsel])


# combine them together
iterate = Iteration([task_train, task_test, task_select], niter, init_folder='share/init', record_file='RECORD')

if os.path.exists('RECORD'):
    iterate.restart()
else:
    iterate.run()
