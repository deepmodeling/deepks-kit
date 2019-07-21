import argparse, os
import numpy as np
import torch
from model import QCNet
from reader import GroupReader
from train import DEVICE, train


def add_bool_arg(parser, name, default=False, help=""):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=name, action='store_false', help=f"opposite to --{name}")
    parser.set_defaults(**{name:default})

def main():
    parser = argparse.ArgumentParser(description="*** Train a model. ***")
    parser.add_argument('-t','--test-path', type=str, nargs = '*',
                        help='the path to testing data .raw files')
    parser.add_argument('-d','--data-path', type=str, nargs = '+',
                        help='the path to training data .raw files')
    parser.add_argument('-n','--neurons-filter', type=int, default=[40, 40, 40], nargs='+',
                        help='the number of neurons in filter net')
    parser.add_argument('-N','--neurons-fit', type=int, default=[100, 100, 100, 100], nargs='+',
                        help='the number of neurons in fitting net')
    parser.add_argument('-s','--shell-sections', type=int, default=None, nargs='*',
                        help='the number of neurons in fitting net')
    parser.add_argument('-b','--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('-e','--num-epoches', type=int, default=3000,
                        help='the number of epoches')
    parser.add_argument('-l','--start-lr', type=float, default=0.005,
                        help='the starting learning rate')
    parser.add_argument('--decay-steps', type=int, default=200,
                        help='the decay steps')
    parser.add_argument('--decay-rate', type=float, default=0.96,
                        help='the decay rate')
    parser.add_argument('-D', '--display_epoch', type=int, default=100,
                        help=r'display training condition in every % epoches')
    parser.add_argument('-S', '--ckpt-file', default='model.pth',
                        help='the file to save parameter file during training in display epoch')
    parser.add_argument('-R', '--restart-file', default=None,
                        help='the restart file to be load before training, if given, would ignore other net settings')
    parser.add_argument('-W','--weight-decay', type=float, default=0.0,
                        help='the factor of weight decay (L2 reg) in training')
    add_bool_arg(parser, "resnet", default=True,
                        help='try using ResNet if two neighboring layers are of the same size')
    args = parser.parse_args()

    g_reader = GroupReader(args.data_path, args.batch_size)
    test_reader = GroupReader(args.test_path, args.batch_size) if args.test_path else None
    if args.restart_file is not None:
        model = QCNet.load(args.restart_file)
    else:
        model = QCNet(args.neurons_filter, args.neurons_fit, 
                        shell_sections=args.shell_sections,
                        e_stat=g_reader.compute_ener_stat(), 
                        use_resnet=args.resnet)
    model = model.double().to(DEVICE)
    train(model, g_reader, args.num_epoches,
            test_reader=test_reader, 
            start_lr=args.start_lr, decay_steps=args.decay_steps, decay_rate=args.decay_rate, weight_decay=args.weight_decay,
            display_epoch=args.display_epoch, ckpt_file=args.ckpt_file)


if __name__ == "__main__":
    main()