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
from deepqc.iterate.iterate import RECORD


def main(*args, **kwargs):
    # pass all arguments to make_iterate and run it
    iterate = make_iterate(*args, **kwargs)
    if os.path.exists('RECORD'):
        iterate.restart()
    else:
        iterate.run()


def cli(args=None):
    parser = argparse.ArgumentParser(
                description="Run the iteration procedure to train a SCF model",
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
    sub_names = ["scf-args", "scf-machine", "train-args", "train-machine",
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