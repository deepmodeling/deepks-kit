import os
import shutil
from glob import glob
from pathlib import Path
import ruamel.yaml as yaml
import numpy as np


QCDIR = os.path.dirname(os.path.realpath(__file__))


# below are argument chekcing utils

def check_list(arg, nullable=True):
    # make sure the argument is a list
    if arg is None:
        if nullable:
            return []
        else:
            raise TypeError("arg cannot be None")
    if not isinstance(arg, (list, tuple, np.ndarray)):
        return [arg]
    return arg


def check_array(arr, nullable=True):
    if arr is None:
        if nullable:
            return arr
        else:
            raise TypeError("arg cannot be None")
    if isinstance(arr, str):
        return load_array(arr)
    else:
        return np.array(arr)


def flat_file_list(file_list, filter_func=lambda p: True):
    # make sure file list contains desired files
    # flat all wildcards and files contains other files (once)
    # if no satisfied files, return empty list
    file_list = check_list(file_list)
    file_list = sorted(sum([glob(p) for p in file_list], []))
    new_list = []
    for p in file_list:
        if filter_func(p):
            new_list.append(p)
        else:
            with open(p) as f:
                sub_list = f.read().splitlines()
                sub_list = sorted(sum([glob(p) for p in sub_list], []))
                new_list.extend(sub_list)
    return new_list


def load_dirs(path_list):
    return flat_file_list(path_list, os.path.isdir)

def load_xyz_files(file_list):
    return flat_file_list(file_list, is_xyz)

def load_sys_paths(sys_list):
    return flat_file_list(sys_list, lambda p: os.path.isdir(p) or is_xyz(p))

def is_xyz(p):
    return os.path.splitext(p)[1] == '.xyz'


# below are file loading utils

def load_yaml(file_path):
    with open(file_path, 'r') as fp:
        res = yaml.safe_load(fp)
    return res


def save_yaml(data, file_path):
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(file_path, 'w') as fp:
        yaml.safe_dump(data, fp)


def load_array(file):
    ext = os.path.splitext(file)[-1]
    if "npy" in ext:
        return np.load(file)
    elif "npz" in ext:
        raise NotImplementedError
    else:
        return np.loadtxt(file)


def parse_xyz(filename):
    with open(filename) as fp:
        natom = int(fp.readline())
        comments = fp.readline().strip()
        atom_str = fp.readlines()
    atom_list = [a.split() for a in atom_str]
    elements = [a[0] for a in atom_list]
    coords = np.array([a[1:] for a in atom_list], dtype=float)
    return natom, comments, elements, coords


# below are path related utils

def get_abs_path(p):
    if p is None:
        return None
    else:
        return Path(p).absolute()


def get_sys_name(p):
    if p.endswith(os.path.sep):
        return p.rstrip(os.path.sep)
    if p.endswith(".xyz"):
        return p[:-4]
    return p


def get_with_prefix(p, base=None, prefer=None, nullable=False):
    """
    Get file path by searching its prefix.
    If `base` is a directory, equals to get "base/p*".
    Otherwise, equals to get "base.p*".
    Only one result will be return. 
    If more than one match, give the first one with suffix in `prefer`.
    """
    if not base:
        base = "./"
    if os.path.isdir(base):
        pattern = os.path.join(base, p)
    else:
        pattern = f"{base.rstrip('.')}.{p}"
    matches = glob(pattern + "*")
    if len(matches) == 1:
        return matches[0]
    prefer = check_list(prefer)
    for suffix in prefer:
        if pattern+suffix in matches:
            return pattern+suffix
    if nullable:
        return None
    raise FileNotFoundError(f"{pattern}* not exists or has more than one matches")

    
def link_file(src, dst, use_abs=False):
    src, dst = Path(src), Path(dst)
    assert src.exists(), f'{src} does not exist'
    src_path = os.path.abspath(src) if use_abs else os.path.relpath(src, dst.parent)
    if not dst.exists():
        if not dst.parent.exists():
            os.makedirs(dst.parent)
        os.symlink(src_path, dst)
    elif not os.path.samefile(src, dst):
        os.remove(dst)
        os.symlink(src_path, dst)


def copy_file(src, dst):
    src, dst = Path(src), Path(dst)
    assert src.exists(), f'{src} does not exist'
    if not dst.exists():
        if not dst.parent.exists():
            os.makedirs(dst.parent)
        shutil.copy2(src, dst)
    elif not os.path.samefile(src, dst):
        os.remove(dst)
        shutil.copy2(src, dst)


def create_dir(dirname, backup=False):
    dirname = Path(dirname)
    if not dirname.exists():
        os.makedirs(dirname)
    elif backup and dirname != Path('.'):
        os.makedirs(dirname.parent, exist_ok=True)
        counter = 0
        bckname = str(dirname) + f'.bck.{counter:03d}'
        while os.path.exists(bckname):
            counter += 1
            bckname = str(dirname) + f'.bck.{counter:03d}'
        dirname.rename(bckname)
        os.makedirs(dirname)
    else:
        assert dirname.is_dir(), f'{dirname} is not a dir'

