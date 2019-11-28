import argparse, os
import numpy as np
import torch
import ruamel_yaml as yaml
from model import QCNet
from reader import GroupReader
from train import DEVICE, train


def load_yaml(file_path):
    with open(file_path, 'r') as fp:
        res = yaml.safe_load(fp)
    return res


def load_sys_paths(path_list):
    new_list = []
    for p in path_list:
        if os.path.isdir(p):
            new_list.append(p)
        else:
            with open(p) as f:
                new_list.extend(f.read().splitlines())
    return new_list


def main():
    parser = argparse.ArgumentParser(description="*** Train a model according to givven input. ***")
    parser.add_argument('input', type=str, 
                        help='the input yaml file for args')
    parser.add_argument('--restart', default=None,
                        help='the restart file to load model from, would ignore model_args if given')
    args = parser.parse_args()
    argdict = load_yaml(args.input)
    
    seed = argdict['seed'] if 'seed' in argdict else np.random.randint(0, 2**32)
    print(f'# using seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_paths = load_sys_paths(argdict['train_paths'])
    print(f'# training with {len(train_paths)} systems')
    g_reader = GroupReader(train_paths, **argdict['data_args'])
    if 'test_paths' in argdict:
        test_paths = load_sys_paths(argdict['test_paths'])
        print(f'# testing with {len(test_paths)} systems')
        test_reader = GroupReader(test_paths, **argdict['data_args'])
    else:
        print('# testing with training set')
        test_reader = None

    if args.restart is not None:
        model = QCNet.load(args.restart)
    else:
        model = QCNet(**argdict['model_args'])
        davg, dstd = g_reader.compute_data_stat()
        model.set_normalization(davg, dstd)
        weight, bias = g_reader.compute_prefitting()
        pf_train = argdict['prefit_trainable'] if 'prefit_trainable' in argdict else False
        model.set_prefitting(weight, bias, trainable=pf_train)
    model = model.double().to(DEVICE)

    train(model, g_reader, test_reader=test_reader, **argdict['train_args'])


if __name__ == "__main__":
    main()
