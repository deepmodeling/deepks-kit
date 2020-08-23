import argparse, os, sys
import numpy as np
import torch
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepqc.train.model import QCNet
from deepqc.train.reader import GroupReader
from deepqc.train.train import DEVICE, train, preprocess
from deepqc.utils import load_yaml, load_sys_dirs


def main(train_paths, test_paths=None,
         restart=None, model_args=None, 
         data_args=None, preprocess_args=None, 
         train_args=None, seed=None, **kwargs):
   
    if seed is None: 
        seed = np.random.randint(0, 2**32)
    print(f'# using seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model_args is None: model_args = {}
    if data_args is None: data_args = {}
    if preprocess_args is None: preprocess_args = {}
    if train_args is None: train_args = {}

    train_paths = load_sys_dirs(train_paths)
    print(f'# training with {len(train_paths)} system(s)')
    g_reader = GroupReader(train_paths, **data_args)
    if test_paths is not None:
        test_paths = load_sys_dirs(test_paths)
        print(f'# testing with {len(test_paths)} system(s)')
        test_reader = GroupReader(test_paths, **data_args)
    else:
        print('# testing with training set')
        test_reader = None

    if restart is not None:
        model = QCNet.load(restart)
    else:
        input_dim = g_reader.ndesc
        if model_args.get("input_dim", input_dim) != input_dim:
            print(f"# `input_dim` in `model_args` does not match data",
                  "({input_dim}).", "Use the one in data.", file=sys.stderr)
        model_args["input_dim"] = input_dim
        model = QCNet(**model_args)
    preprocess(model, g_reader, **preprocess_args)
    model = model.double().to(DEVICE)

    train(model, g_reader, test_reader=test_reader, **train_args)


def cli():
    parser = argparse.ArgumentParser(
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
    parser.add_argument('-S', '--seed', type=int,
                        help='use specified seed in initialization and training')
    args = parser.parse_args()
    
    if hasattr(args, "input"):
        argdict = load_yaml(args.input)
        del args.input
        argdict.update(vars(args))
    else:
        argdict = vars(args)

    main(**argdict)


if __name__ == "__main__":
    cli()