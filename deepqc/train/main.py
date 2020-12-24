import os, sys
import numpy as np
import torch
try:
    import deepqc
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepqc.train.model import CorrNet
from deepqc.train.reader import GroupReader
from deepqc.train.train import train, preprocess
from deepqc.utils import load_yaml, load_dirs


def main(train_paths, test_paths=None,
         restart=None, ckpt_file=None, 
         model_args=None, data_args=None, 
         preprocess_args=None, train_args=None, 
         seed=None, device=None):
   
    if seed is None: 
        seed = np.random.randint(0, 2**32)
    print(f'# using seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model_args is None: model_args = {}
    if data_args is None: data_args = {}
    if preprocess_args is None: preprocess_args = {}
    if train_args is None: train_args = {}
    if ckpt_file is not None:
        train_args["ckpt_file"] = ckpt_file
    if device is not None:
        train_args["device"] = device

    train_paths = load_dirs(train_paths)
    print(f'# training with {len(train_paths)} system(s)')
    g_reader = GroupReader(train_paths, **data_args)
    if test_paths is not None:
        test_paths = load_dirs(test_paths)
        print(f'# testing with {len(test_paths)} system(s)')
        test_reader = GroupReader(test_paths, **data_args)
    else:
        print('# testing with training set')
        test_reader = None

    if restart is not None:
        model = CorrNet.load(restart)
    else:
        input_dim = g_reader.ndesc
        if model_args.get("input_dim", input_dim) != input_dim:
            print(f"# `input_dim` in `model_args` does not match data",
                  "({input_dim}).", "Use the one in data.", file=sys.stderr)
        model_args["input_dim"] = input_dim
        model = CorrNet(**model_args).double()
        
    preprocess(model, g_reader, **preprocess_args)
    train(model, g_reader, test_reader=test_reader, **train_args)


if __name__ == "__main__":
    from deepqc.main import train_cli as cli
    cli()