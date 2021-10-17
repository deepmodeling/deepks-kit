import os
import sys
import numpy as np
try:
    import deepks
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepks.utils import copy_file, link_file
from deepks.utils import load_yaml, save_yaml
from deepks.utils import load_sys_paths
from deepks.utils import load_basis, save_basis
from deepks.task.workflow import Sequence, Iteration
from deepks.iterate.template import make_scf, make_train
from deepks.iterate.template_abacus import make_scf_abacus  #caoyu add 2021-07-22 
from deepks.iterate.template_abacus import DEFAULT_SCF_ARGS_ABACUS


# args not specified here may cause error
DEFAULT_SCF_MACHINE = {
    "sub_size": 1, # how many systems is put in one task (folder)
    "sub_res": None, # the resources for sub step when ingroup_parallel > 1
    "group_size": 1, # how many tasks are submitted in one job
    "ingroup_parallel": 1, #how many tasks can run at same time in one job
    "dispatcher": None, # use default lazy-local slurm defined in task.py
    "resources": None, # use default 10 core defined in templete.py
    "python": "python" # use current python in path
}

# args not specified here may cause error
DEFAULT_TRN_MACHINE = {
    "dispatcher": None, # use default lazy-local slurm defined in task.py
    "resources": None, # use default 10 core defined in templete.py
    "python": "python" # use current python in path
}

SCF_ARGS_NAME = "scf_input.yaml"
SCF_ARGS_NAME_ABACUS="scf_abacus.yaml"   #for abacus, caoyu add 2021-07-26
TRN_ARGS_NAME = "train_input.yaml"
INIT_SCF_NAME = "init_scf.yaml"
INIT_TRN_NAME = "init_train.yaml"

DATA_TRAIN = "data_train"
DATA_TEST  = "data_test"
MODEL_FILE = "model.pth"
PROJ_BASIS = "proj_basis.npz"

SCF_STEP_DIR = "00.scf"
TRN_STEP_DIR = "01.train"

RECORD = "RECORD"

SYS_TRAIN = "systems_train"
SYS_TEST = "systems_test"
DEFAULT_TRAIN = "systems_train.raw"
DEFAULT_TEST = "systems_test.raw"


def assert_exist(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No required file or directory: {path}")


def check_share_folder(data, name, share_folder="share"):
    # save data to share_folder/name. 
    # if data is None or False, do nothing, return None
    # otherwise, return name, and do one of the following:
    #   if data is True, check the existence in share.
    #   if data is a file name, copy it to share.
    #   if data is a dict, save it as an yaml file in share.
    #   otherwise, throw an error
    if not data:
        return None
    dst_name = os.path.join(share_folder, name)
    if data is True:
        assert_exist(dst_name)
        return name
    elif isinstance(data, str) and os.path.exists(data):
        copy_file(data, dst_name)
        return name
    elif isinstance(data, dict):
        save_yaml(data, dst_name)
        return name
    else:
        raise ValueError(f"Invalid argument: {data}")


def check_arg_dict(data, default, strict=True):
    if data is None:
        data = {}
    if isinstance(data, str):
        data = load_yaml(data)
    allowed = {k:v for k,v in data.items() if k in default}
    outside = {k:v for k,v in data.items() if k not in default}
    if outside:
        print(f"following ars are not in the default list: {list(outside.keys())}"
              +"and would be discarded" if strict else "but kept", file=sys.stderr)
    if strict:
        return {**default, **allowed}
    else:
        return {**default, **data}


def collect_systems(systems, folder=None):
    # check all systems have different basename
    # if there's duplicate, concat its dirname into the basename sep by a "."
    # then collect all systems into `folder` by symlink
    sys_list = [os.path.abspath(s) for s in load_sys_paths(systems)]
    parents, bases = map(list, zip(*[os.path.split(s.rstrip(os.path.sep)) 
                                        for s in sys_list]))
    dups = range(len(sys_list))
    while True:
        count_dict = {bases[i]:[] for i in dups}
        for i in dups:
            count_dict[bases[i]].append(i)
        dup_dict = {k:v for k,v in count_dict.items() if len(v)>1}
        if not dup_dict:
            break
        dups = sum(dup_dict.values(), [])
        if all(parents[i] in ("/", "") for i in dups):
            print("System list have duplicated terms, index:", dups, file=sys.stderr)
            break
        for di in dups:
            if parents[di] in ("/", ""):
                continue
            newp, newb = os.path.split(parents[di])
            parents[di] = newp
            bases[di] = f"{newb}.{bases[di]}"
    if folder is None:
        return bases
    targets = [os.path.join(folder, b) for b in bases]
    for s, t in zip(sys_list, targets):
        link_file(s, t, use_abs=True)
    return targets


def make_iterate(systems_train=None, systems_test=None, n_iter=0, 
                 *, proj_basis=None, workdir=".", share_folder="share",
                 scf_input=True, scf_machine=None,
                 train_input=True, train_machine=None,
                 init_model=False, init_scf=True, init_train=True,
                 init_scf_machine=None, init_train_machine=None,
                 cleanup=False, strict=True, 
                 use_abacus=False, scf_abacus=None):#caoyu add 2021-07-22
    r"""
    Make a `Workflow` to do the iterative training procedure.

    The procedure will be conducted in `workdir` for `n_iter` iterations.
    Each iteration of the procedure is done in sub-folder ``iter.XX``, 
    which further containes two sub-folders, ``00.scf`` and ``01.train``.
    The `Workflow` is only created but not executed.

    Parameters
    ----------
    systems_train: str or list of str, optional
        System paths used as training set in the procedure. These paths 
        can refer to systems or a file that contains multiple system paths.
        Systems must be .xyz files or folders contains .npy files.
        If not given, use ``$share_folder/systems_train.raw`` as default.
    systems_test: str or list of str, optional
        System paths used as testing (or validation) set in the procedure. 
        The format is same as `systems_train`. If not given, use the last
        system in the training set as testing system.
    n_iter: int, optional
        The number of iterations to do. Default is 0.
    proj_basis: str, optional
        The basis set used to project the density matrix onto. 
        Can be a `.npz` file specifying the coefficients in pyscf's format.
        If not given, use the default basis.
    workdir: str, optional
        The working directory. Default is current directory (`.`).
    share_folder: str, optional
        The folder to store shared files in the iteration, including
        ``scf_input.yaml``, ``train_input.yaml``, and possibly files for
        initialization. Default is ``share``.
    scf_input: bool or str or dict, optional
        Arguments used to specify the SCF calculation. If given `None` or
        `False`, bypass the checking and use program default (unreliable). 
        Otherwise, the arguments would be saved as a YAML file at 
        ``$share_folder/scf_input.yaml`` and used for SCF calculation. 
        Default is `True`, which will check and use the existing file.
        If given a string of file path, copy the corresponding file into 
        target location. If given a dict, dump it into the target file.
    scf_machine: str or dict, optional
        Arguments used to specify the job settings of SCF calculation,
        including submitting method, resources, group size, etc..
        If given a string of file path, load that file as a dict using 
        YAML format. If not given, using program default setup.
    train_input: bool or str or dict, optional 
        Arguments used to specify the training of neural network. 
        It follows the same rule as `scf_input`, only that the target 
        location is ``$share_folder/train_input.yaml``.
    train_machine: str or dict, optional 
        Arguments used to specify the job settings of NN training. 
        It Follows the same rule as `scf_machine`, but without group.
    init_model: bool or str, optional 
        Decide whether to use an existing model as the starting point.
        If set to `False` (default), use `init_scf` and `init_train` 
        to run an extra initialization iteration in folder ``iter.init``. 
        If set to `True`, look for a model at ``$share_folder/init/model.pth``.
        If given a string of path, copy that file into target location.
    init_scf: bool or str or dict, optional 
        Similar to `scf_input` but used for init calculation. The target
        location is ``$share_folder/init_scf.yaml``. Defaults to True.
    init_scf_machine: str or dict, optional
        If specified, use different machine settings for init scf jobs.
    init_train: bool or str or dict, optional 
        Similar to `train_input` but used for init calculation. The target
        location is ``$share_folder/init_train.yaml``. Defaults to True.
    init_train_machine: str or dict, optional
        If specified, use different machine settings for init training job.
    cleanup: bool, optional 
        Whether to remove job files during calculation, 
        such as ``slurm-*.out`` and ``err``. Defaults to False.
    strict: bool, optional 
        Whether to allow additional arguments to be passed to task constructor,
        through `scf_machine` and `train_machine`. Defaults to True.

    Returns
    -------
    iterate: Iteration (subclass of Workflow)
        An instance of workflow that can be executed by `iterate.run()`.
    
    Raises
    ------
    FileNotFoundError
        Raise an Error when the system or argument files are required but 
        not found in the share folder.
    """
    # check share folder contains required data
    # and collect the systems into share folder
    if systems_train is None: # load default training systems
        default_train = os.path.join(share_folder, DEFAULT_TRAIN)
        assert_exist(default_train) # must have training systems.
        systems_train = default_train
    systems_train = collect_systems(systems_train, os.path.join(share_folder, SYS_TRAIN))
    # check test systems 
    if systems_test is None: # try to load default testing systems
        default_test = os.path.join(share_folder, DEFAULT_TEST)
        if os.path.exists(default_test): # if exists then use it
            systems_test = default_test
        else: # if empty use last one of training system
            systems_test = systems_train[-1]
    systems_test = collect_systems(systems_test, os.path.join(share_folder, SYS_TEST))
    # check share folder contains required yaml file
    scf_args_name = check_share_folder(scf_input, SCF_ARGS_NAME, share_folder)
    train_args_name = check_share_folder(train_input, TRN_ARGS_NAME, share_folder)
    # check required machine parameters
    scf_machine = check_arg_dict(scf_machine, DEFAULT_SCF_MACHINE, strict)
    train_machine = check_arg_dict(train_machine, DEFAULT_TRN_MACHINE, strict)

    # make tasks
    if use_abacus:  #caoyu add 2021-07-22
        scf_abacus_name = check_share_folder(scf_abacus, SCF_ARGS_NAME_ABACUS, share_folder)
        scf_abacus = check_arg_dict(scf_abacus, DEFAULT_SCF_ARGS_ABACUS, strict)
        scf_abacus = dict(scf_abacus, **scf_machine)
        scf_step = make_scf_abacus(systems_train=systems_train, systems_test=systems_test,
            train_dump=DATA_TRAIN, test_dump=DATA_TEST, no_model=False,
            workdir=SCF_STEP_DIR, share_folder=share_folder,
            cleanup=cleanup, **scf_abacus)
        proj_basis=None     # discussion needed
    else:
        # handle projection basis
        if proj_basis is not None:
            save_basis(os.path.join(share_folder, PROJ_BASIS), load_basis(proj_basis))
            proj_basis = PROJ_BASIS
        scf_step = make_scf(
            systems_train=systems_train, systems_test=systems_test,
            train_dump=DATA_TRAIN, test_dump=DATA_TEST, no_model=False,
            workdir=SCF_STEP_DIR, share_folder=share_folder,
            source_arg=scf_args_name, source_model=MODEL_FILE,
            source_pbasis=proj_basis, cleanup=cleanup, **scf_machine
        )
    train_step = make_train(
        source_train=DATA_TRAIN, source_test=DATA_TEST,
        restart=True, source_model=MODEL_FILE, save_model=MODEL_FILE, 
        source_pbasis=proj_basis, source_arg=train_args_name, 
        workdir=TRN_STEP_DIR, share_folder=share_folder,
        cleanup=cleanup, **train_machine
    )
    per_iter = Sequence([scf_step, train_step])
    iterate = Iteration(per_iter, n_iter, 
                        workdir=".", record_file=os.path.join(workdir, RECORD))

    # make init
    if init_model: # if set true or give str, check share/init/model.pth
        init_folder=os.path.join(share_folder, "init")
        check_share_folder(init_model, MODEL_FILE, init_folder)
        iterate.set_init_folder(init_folder)
    elif init_scf or init_train: # otherwise, make an init iteration to train the first model
        init_scf_machine = (check_arg_dict(init_scf_machine, DEFAULT_SCF_MACHINE, strict)
            if init_scf_machine is not None else scf_machine)
        if use_abacus:  #caoyu add 2021-07-22
            scf_init = make_scf_abacus(
                systems_train=systems_train, systems_test=systems_test,
                train_dump=DATA_TRAIN, test_dump=DATA_TEST, no_model=True,
                workdir=SCF_STEP_DIR, share_folder=share_folder, model_file=None, 
                cleanup=cleanup, **scf_abacus
            )
        else:
            init_scf_name = check_share_folder(init_scf, INIT_SCF_NAME, share_folder)
            scf_init = make_scf(
                systems_train=systems_train, systems_test=systems_test,
                train_dump=DATA_TRAIN, test_dump=DATA_TEST, no_model=True,
                workdir=SCF_STEP_DIR, share_folder=share_folder,
                source_arg=init_scf_name, source_model=None, source_pbasis=proj_basis,
                cleanup=cleanup, **scf_machine
            )
        init_train_name = check_share_folder(init_train, INIT_TRN_NAME, share_folder)
        init_train_machine = (check_arg_dict(init_train_machine, DEFAULT_SCF_MACHINE, strict)
            if init_train_machine is not None else train_machine)
        train_init = make_train(
            source_train=DATA_TRAIN, source_test=DATA_TEST,
            restart=False, source_model=MODEL_FILE, save_model=MODEL_FILE, 
            source_pbasis=proj_basis, source_arg=init_train_name, 
            workdir=TRN_STEP_DIR, share_folder=share_folder,
            cleanup=cleanup, **train_machine
        )
        init_iter = Sequence([scf_init, train_init], workdir="iter.init")
        iterate.prepend(init_iter)
    return iterate


def main(*args, **kwargs):
    r"""
    Make a `Workflow` to do the iterative training procedure and run it.

    The parameters are the same as `make_iterate` but the jobs wil be run.
    If ``$workdir/RECORD`` exists, the procedure will try to restart.
    The procedure will be conducted in `workdir` for `n_iter` iterations.
    Each iteration of the procedure is done in sub-folder ``iter.XX``, 
    which further containes two sub-folders, ``00.scf`` and ``01.train``.

    See `make_iterate` for detailed parameters.
    """
    # pass all arguments to make_iterate and run it
    iterate = make_iterate(*args, **kwargs)
    if os.path.exists(iterate.record_file):
        iterate.restart()
    else:
        iterate.run()


if __name__ == "__main__":
    from deepks.main import iter_cli as cli
    cli()