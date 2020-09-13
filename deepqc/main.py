import os
import sys
import argparse
try:
    import deepqc
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from deepqc.utils import load_yaml


def main_cli(args=None):
    parser = argparse.ArgumentParser(
                prog="deepqc",
                description="A program to generate accurate energy functionals.")
    parser.add_argument("command", 
                        help="specify the sub-command to run, possible choices: "
                             "train, test, scf, stat, iterate")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                        help="arguments to be passed to the sub-command")

    args = parser.parse_args(args)

    # sepatate all sub_cli to make them useable independently 
    if args.command.upper() == "TRAIN":
        sub_cli = train_cli
    elif args.command.upper() == "TEST":
        sub_cli = test_cli
    elif args.command.upper() == "SCF":
        sub_cli = scf_cli
    elif args.command.upper() == "STAT":
        sub_cli = stat_cli
    elif args.command.upper().startswith("ITER"):
        sub_cli = iter_cli
    else:
        return ValueError(f"unsupported sub-command: {args.command}")
    
    sub_cli(args.args)


def train_cli(args=None):
    parser = argparse.ArgumentParser(
                prog="deepqc train",
                description="Train a model according to given input.",
                argument_default=argparse.SUPPRESS)
    parser.add_argument('input', type=str, nargs="?",
                        help='the input yaml file for args')
    parser.add_argument('-r', '--restart',
                        help='the restart file to load model from, would ignore model_args if given')
    parser.add_argument('-d', '--train-paths', nargs="*",
                        help='paths to the folders of training data')
    parser.add_argument('-t', '--test-paths', nargs="*",
                        help='paths to the folders of testing data')
    parser.add_argument('-o', '--ckpt-file',
                        help='file to save the model parameters, default: model.pth')
    parser.add_argument('-S', '--seed', type=int,
                        help='use specified seed in initialization and training')
    parser.add_argument("-D", "--device",
                        help="device name used in training the model")    
    args = parser.parse_args(args)
    
    if hasattr(args, "input"):
        argdict = load_yaml(args.input)
        del args.input
        argdict.update(vars(args))
    else:
        argdict = vars(args)

    from deepqc.train.main import main
    main(**argdict)


def test_cli(args=None):
    parser = argparse.ArgumentParser(
                prog="deepqc test",
                description="Test a model with given data (Not SCF).",
                argument_default=argparse.SUPPRESS)
    parser.add_argument("input", nargs="?",
                        help='the input yaml file used for training')
    parser.add_argument("-d", "--data-paths", type=str, nargs='+',
                        help="the paths to data folders containing .npy files for test")
    parser.add_argument("-m", "--model-file", type=str, nargs='+',
                        help="the dumped model file to test")
    parser.add_argument("-o", "--output-prefix", type=str,
                        help=r"the prefix of output file, would wite into file %%prefix.%%sysidx.out")
    parser.add_argument("-E", "--e-name", type=str,
                        help="the name of energy file to be read (no .npy extension)")
    parser.add_argument("-D", "--d-name", type=str, nargs="+",
                        help="the name of descriptor file(s) to be read (no .npy extension)")
    parser.add_argument("-G", "--group", action='store_true',
                        help="group test results for all systems")
    args = parser.parse_args(args)

    if hasattr(args, "input"):
        rawdict = load_yaml(args.input)
        del args.input
        argdict = {}
        if "ckpt_file" in rawdict["train_args"]:
            argdict["model_file"] = rawdict["train_args"]["ckpt_file"]
        if "e_name" in rawdict["data_args"]:
            argdict["e_name"] = rawdict["data_args"]["e_name"]
        if "d_name" in rawdict["data_args"]:
            argdict["d_name"] = rawdict["data_args"]["d_name"]
        if "test_paths" in rawdict:
            argdict["data_paths"] = rawdict["test_paths"]
        argdict.update(vars(args))
    else:
        argdict = vars(args)

    from deepqc.train.test import main
    main(**argdict)


def scf_cli(args=None):
    parser = argparse.ArgumentParser(
                prog="deepqc scf",
                description="Calculate and save SCF results using given model.",
                argument_default=argparse.SUPPRESS)
    parser.add_argument("input", nargs="?",
                        help='the input yaml file for args')
    parser.add_argument("-s", "--systems", nargs="*",
                        help="input molecule systems, can be xyz files or folders with npy data")
    parser.add_argument("-m", "--model-file",
                        help="file of the trained model")
    parser.add_argument("-B", "--basis",
                        help="basis set used to solve the model") 
    parser.add_argument("-P", "--proj_basis",
                        help="basis set used to project dm, must match with model")   
    parser.add_argument("-D", "--device",
                        help="device name used in nn model inference")               
    parser.add_argument("-d", "--dump-dir",
                        help="dir of dumped files")
    parser.add_argument("-F", "--dump-fields", nargs="*",
                        help="fields to be dumped into the folder") 
    group0 = parser.add_mutually_exclusive_group()   
    group0.add_argument("-G", "--group", action='store_true', dest="group",
                        help="group results for all systems, only works for same system")
    group0.add_argument("-NG", "--no-group", action='store_false', dest="group",
                        help="Do not group results for different systems (default behavior)")
    parser.add_argument("-v", "--verbose", type=int, choices=range(0,10),
                        help="output calculation information")
    parser.add_argument("-X", "--scf-xc",
                        help="base xc functional used in scf equation, default is HF")        
    parser.add_argument("--scf-conv-tol", type=float,
                        help="converge threshold of scf iteration")
    parser.add_argument("--scf-conv-tol-grad", type=float,
                        help="gradient converge threshold of scf iteration")
    parser.add_argument("--scf-max-cycle", type=int,
                        help="max number of scf iteration cycles")
    parser.add_argument("--scf-diis-space", type=int,
                        help="subspace dimension used in diis mixing")
    parser.add_argument("--scf-level-shift", type=float,
                        help="level shift used in scf calculation")

    args = parser.parse_args(args)

    scf_args={}
    for k, v in vars(args).copy().items():
        if k.startswith("scf_"):
            scf_args[k[4:]] = v
            delattr(args, k)

    if hasattr(args, "input"):
        argdict = load_yaml(args.input)
        del args.input
        argdict.update(vars(args))
        argdict["scf_args"].update(scf_args)
    else:
        argdict = vars(args)
        argdict["scf_args"] = scf_args

    from deepqc.scf.main import main
    main(**argdict)


def stat_cli(args=None):
    parser = argparse.ArgumentParser(
                prog="deepqc stat",
                description="Print the stat of SCF results.",
                argument_default=argparse.SUPPRESS)
    parser.add_argument("input", nargs="?",
                        help='the input yaml file used for SCF calculation')
    parser.add_argument("-s", "--systems", nargs="*",
                        help='system paths used as training set (i.e. calculate shift)')
    parser.add_argument("-d", "--dump-dir",
                        help="directory used to save SCF results of training systems")
    parser.add_argument("-ts", "--test-sys", nargs="*",
                        help='system paths used as testing set (i.e. not calculate shift)')
    parser.add_argument("-td", "--test-dump",
                        help="directory used to save SCF results of testing systems")
    parser.add_argument("-G", "--group", action='store_true',
                        help="if set, assume results are grouped")
    parser.add_argument("-NC", action="store_false", dest="with_conv",
                        help="do not print convergence results")
    parser.add_argument("-NE", action="store_false", dest="with_e",
                        help="do not print energy results")
    parser.add_argument("-NF", action="store_false", dest="with_f",
                        help="do not print force results")
    parser.add_argument("--e-name",
                        help="name of the energy file (no extension)")
    parser.add_argument("--f-name",
                        help="name of the force file (no extension)")
    args = parser.parse_args(args)

    if hasattr(args, "input"):
        rawdict = load_yaml(args.input)
        del args.input
        argdict = {fd: rawdict[fd]
                     for fd in ("systems", "dump_dir", "group")
                     if fd in rawdict}
        argdict.update(vars(args))
    else:
        argdict = vars(args)

    from deepqc.scf.tools import print_stat
    print_stat(**argdict)


def iter_cli(args=None):
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

    from deepqc.iterate.main import main
    main(**argdict)


if __name__ == "__main__":
    main_cli()