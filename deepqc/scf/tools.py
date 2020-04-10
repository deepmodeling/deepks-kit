import os
import sys
import glob
import numpy as np
import shutil
from pathlib import Path


def get_array(arr):
    if isinstance(arr, str):
        ext = os.path.splitext(arr)[-1]
        if "npy" in ext:
            return np.load(arr)
        elif "npz" in ext:
            raise NotImplementedError
        else:
            return np.loadtxt(arr)
    else:
        return np.array(arr)


def print_stat(err, conv=None, train_idx=None, test_idx=None):
    err = np.array(err).reshape(-1)
    nsys = err.shape[0]
    if conv is not None:
        assert len(conv) == nsys
        print(f'converged calculation: {np.sum(conv)} / {nsys} = {np.mean(conv):.3f}')
    print(f'mean error: {err.mean()}')
    print(f'mean absolute error: {np.abs(err).mean()}')
    if train_idx is not None:
        print(f'mean absolute error after shift: {np.abs(err - err[train_idx].mean()).mean()}')
        print(f'  training: {np.abs(err[train_idx] - err[train_idx].mean()).mean()}')
        if test_idx is None:
            test_idx = np.setdiff1d(np.arange(nsys), train_idx, assume_unique=True)
        print(f'  testing: {np.abs(err[test_idx] - err[train_idx].mean()).mean()}')


def collect_data(train_idx, test_idx=None, 
                 sys_dir="results", ene_ref="e_ref.npy", 
                 dump_dir=".", verbose=True):
    erefs = get_array(ene_ref).reshape(-1)
    nsys = erefs.shape[0]
    if nsys == 1 and "e_cf.npy" in os.listdir(sys_dir):
        systems = [os.path.abspath(sys_dir)]
    else:
        systems = sorted(map(os.path.abspath, glob.glob(f"{sys_dir}/*")))
    assert nsys == len(systems)

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

    if test_idx is None:
        test_idx = np.setdiff1d(np.arange(nsys), train_idx, assume_unique=True)
    if verbose:
        print_stat(err, convs, train_idx, test_idx)
    
    np.savetxt(f'{dump_dir}/train_paths.raw', np.array(systems)[train_idx], fmt='%s')
    np.savetxt(f'{dump_dir}/test_paths.raw', np.array(systems)[test_idx], fmt='%s')
    np.savetxt(f'{dump_dir}/e_result.out', np.stack([erefs, ecfs], axis=-1), header="real pred")


def collect_data_grouped(train_idx, test_idx=None, 
                         sys_dir="results", ene_ref="e_ref.npy", 
                         dump_dir=".", verbose=True):
    eref = get_array(ene_ref).reshape(-1, 1)
    nmol = eref.shape[0]
    ecf = np.load(f'{sys_dir}/e_cf.npy').reshape(-1, 1)
    assert ecf.shape[0] == nmol
    ehf = np.load(f'{sys_dir}/e_hf.npy')
    np.save(f'{sys_dir}/e_cc.npy', eref - ehf)

    err = eref - ecf
    conv = np.load(f'{sys_dir}/conv.npy').reshape(-1)
    if test_idx is None:
        test_idx = np.setdiff1d(np.arange(nmol), train_idx, assume_unique=True)
    if verbose:
        print_stat(err.reshape(-1), conv.reshape(-1), train_idx, test_idx)
    
    dd = ['dm_eig.npy', 'e_cc.npy']
    os.makedirs(f'{dump_dir}/train', exist_ok=True)
    os.makedirs(f'{dump_dir}/test', exist_ok=True)
    for d in dd:
        np.save(f"{dump_dir}/train/{d}", np.load(f'{sys_dir}/{d}')[train_idx])
    for d in dd:
        np.save(f"{dump_dir}/test/{d}", np.load(f'{sys_dir}/{d}')[test_idx])
    shutil.copy(f'{sys_dir}/system.raw', f'{dump_dir}/train')
    shutil.copy(f'{sys_dir}/system.raw', f'{dump_dir}/test')
    np.savetxt(f'{dump_dir}/train_paths.raw', [os.path.abspath(f'{dump_dir}/train')], fmt='%s')
    np.savetxt(f'{dump_dir}/test_paths.raw', [os.path.abspath(f'{dump_dir}/test')], fmt='%s')
    # Path(f'{dump_dir}/train_paths.raw').write_text(str(Path(f'{dump_dir}/train').absolute()))
    # Path(f'{dump_dir}/test_paths.raw').write_text(str(Path(f'{dump_dir}/test').absolute()))

