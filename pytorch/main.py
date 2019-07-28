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

def main():
    parser = argparse.ArgumentParser(description="*** Train a model according to givven input. ***")
    parser.add_argument('input', type=str, 
                        help='the input yaml file for args')
    parser.add_argument('--restart', default=None,
                        help='the restart file to load model from, would ignore model_args if given')
    args = parser.parse_args()
    argdict = load_yaml(args.input)

    data_args = argdict['data_args']
    g_reader = GroupReader(data_args['train_path'], data_args['batch_size'])
    test_reader = GroupReader(data_args['test_path'], 1) if 'test_path' in data_args else None
    
    if args.restart is not None:
        model = QCNet.load(args.restart)
    else:
        model = QCNet(**argdict['model_args'], e_stat=g_reader.compute_ener_stat())
    model = model.double().to(DEVICE)

    train(model, g_reader, **argdict['train_args'])


if __name__ == "__main__":
    main()