import os
import sys
import argparse
import numpy as np
try:
    import deepqc
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepqc.utils import load_yaml
from deepqc.iterate.iterate import make_iterate


def main(*args, **kwargs):
    r"""
    Make a `Workflow` to do the iterative training procedure and run it.

    The parameters are the same as `make_iterate` but the jobs wil be run.
    If ``$workdir/RECORD`` exists, the procedure will try to restart.
    The procedure will be conducted in `workdir` for `n_iter` iterations.
    Each iteration of the procedure is done in sub-folder ``iter.XX``, 
    which further containes two sub-folders, ``00.scf`` and ``01.train``.

    Parameters
    ----------
    systems_train: optional str or list of str 
        System paths used as training set in the procedure. These paths 
        can refer to systems or a file that contains multiple system paths.
        Systems must be .xyz files or folder contains .npy files.
        If given `None`, use ``$share_folder/systems_train.raw`` as default.
    systems_test: optional str or list of str
        System paths used as testing (or validation) set in the procedure. 
        The format is same as `systems_train`. If given `None`, use the last
        system in the training set as testing system.
    n_iter: int
        The number of iterations to do. Default is 5.
    workdir: str
        The working directory. Default is current directory (`.`).
    share_folder: str
        The folder to store shared files in the iteration, including
        ``scf_input.yaml``, ``train_input.yaml``, and possibly files for
        initialization. Default is ``share``.
    scf_input: bool or str or dict
        Arguments used to specify the SCF calculation. If given `None` or 
        `False`, use program default (unreliable). Otherwise, the arguments 
        would be saved as a YAML file at ``$share_folder/scf_input.yaml``
        and used for SCF calculation. If given `True`, use the existing file.
        If given a string of file path, copy the corresponding file into 
        target location. If given a dict, dump it into the target file.
    scf_machine: optional str or dict
        Arguments used to specify the job settings of SCF calculation,
        including submitting method, resources, group size, etc..
        If given a string of file path, load that file as a dict using 
        YAML format. If `strict` is set to false, additional arguments
        can be passed to `Task` constructor to do more customization.
    train_input: bool or str or dict
        Arguments used to specify the training of neural network. 
        It follows the same rule as `scf_input`, only that the target 
        location is ``$share_folder/train_input.yaml``.
    train_machine: optional str or dict
        Arguments used to specify the job settings of NN training. 
        It Follows the same rule as `scf_machine`, but without group.
    init_model: bool or str
        Decide whether to use an existing model as the starting point.
        If set to `True`, look for a model at ``$share_folder/init/model.pth``
        If set to `False`, use `init_scf` and `init_train` to run an
        extra initialization iteration in folder ``iter.init``. 
        If given a string of path, copy that file into target location.
    init_scf: bool or str or dict
        Similar to `scf_input` but used for init calculation. The target
        location is ``$share_folder/init_scf.yaml``.
    init_train: bool or str or dict
        Similar to `train_input` but used for init calculation. The target
        location is ``$share_folder/init_train.yaml``.
    cleanup: bool
        Whether to remove job files during calculation, such as `slurm-*.out`.
    strict: bool
        Whether to allow additional arguments to be passed to task constructor.

    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        Raise an Error when the system or argument files are required but 
        not found in the share folder.
    """
    # pass all arguments to make_iterate and run it
    iterate = make_iterate(*args, **kwargs)
    if os.path.exists(iterate.record_file):
        iterate.restart()
    else:
        iterate.run()


def cli(args=None):
    parser = argparse.ArgumentParser(
                prog="deepqc iterate",
                description="Run the iteration procedure to train a SCF model.",
                argument_default=argparse.SUPPRESS)
    parser.add_argument("argfile", nargs="*", default=[],
                        help='the input yaml file for args, '
                             'if more than one, the latter has higher priority')
    parser.add_argument("-s", "--systems-train", nargs="*",
                        help='systems for training, '
                             'can be xyz files or folders with npy data')
    parser.add_argument("-t", "--systems-test", nargs="*",
                        help='systems for training, '
                             'can be xyz files or folders with npy data')
    parser.add_argument("-n", "--n-iter", type=int,
                        help='the number of iterations to run')
    parser.add_argument("--workdir",
                        help='working directory, default is current directory')
    parser.add_argument("--share-folder", 
                        help='folder to store share files, default is "share"')
    parser.add_argument("--cleanup", action="store_true", dest="cleanup",
                        help='if set, clean up files used for job dispatching')
    parser.add_argument("--no-strict", action="store_false", dest="strict",
                        help='if set, allow other arguments to be passed to task')
    # allow cli specified argument files
    sub_names = ["scf-input", "scf-machine", "train-input", "train-machine",
                 "init-model", "init-scf", "init-train"]
    for name in sub_names:
        parser.add_argument(f"--{name}",
            help='if specified, subsitude the original arguments with given file')
    
    args = parser.parse_args(args)
    argdict = {}
    for fl in args.argfile:
        argdict.update(load_yaml(fl))
    del args.argfile
    argdict.update(vars(args))

    main(**argdict)


if __name__ == "__main__":
    cli()