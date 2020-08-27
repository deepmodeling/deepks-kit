import argparse, os
import numpy as np
import torch
import torch.nn as nn
import os
try:
    import deepqc
except ImportError as e:
    import sys
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepqc.train.model import QCNet
from deepqc.train.reader import GroupReader
from deepqc.utils import load_yaml, load_sys_dirs, check_list


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, g_reader, dump_prefix="test", group=False):
    loss_fn=nn.MSELoss()
    label_list = []
    pred_list = []

    for i in range(g_reader.nsystems):
        sample = g_reader.sample_all(i)
        nframes = sample[0].shape[0]
        label, data, *_ = [d.to(DEVICE) for d in sample]
        pred = model(data)
        error = torch.sqrt(loss_fn(pred, label))

        error_np = error.item()
        label_np = label.cpu().numpy().reshape(nframes, -1).sum(axis=1)
        pred_np = pred.detach().cpu().numpy().reshape(nframes, -1).sum(axis=1)
        error_l1 = np.mean(np.abs(label_np - pred_np))
        label_list.append(label_np)
        pred_list.append(pred_np)

        if not group and dump_prefix is not None:
            nd = max(len(str(g_reader.nsystems)), 2)
            dump_res = np.stack([label_np, pred_np], axis=1)
            header = f"{g_reader.path_list[i]}\nmean l1 error: {error_l1}\nmean l2 error: {error_np}\nreal_ene  pred_ene"
            filename = f"{dump_prefix}.{i:0{nd}}.out"
            np.savetxt(filename, dump_res, header=header)
            # print(f"system {i} finished")

    all_label = np.concatenate(label_list, axis=0)
    all_pred = np.concatenate(pred_list, axis=0)
    all_err_l1 = np.mean(np.abs(all_label - all_pred))
    all_err_l2 = np.sqrt(np.mean((all_label - all_pred) ** 2))
    info = f"all systems mean l1 error: {all_err_l1}\nall systems mean l2 error: {all_err_l2}"
    print(info)
    if dump_prefix is not None and group:
        np.savetxt(f"{dump_prefix}.out", np.stack([all_label, all_pred], axis=1), 
                   header=info + "\nreal_ene  pred_ene")
    return all_err_l1, all_err_l2


def main(data_paths, model_file="model.pth", 
         output_prefix='test', group=False,
         e_name='l_e_delta', d_name=['dm_eig']):
    data_paths = load_sys_dirs(data_paths)
    g_reader = GroupReader(data_paths, e_name=e_name, d_name=d_name)
    model_file = check_list(model_file)
    for f in model_file:
        print(f)
        p = os.path.dirname(f)
        model = QCNet.load(f).double().to(DEVICE)
        dump = os.path.join(p, output_prefix)
        dir_name = os.path.dirname(dump)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        test(model, g_reader, dump_prefix=dump, group=group)


def cli():
    parser = argparse.ArgumentParser(
                description="Test a model with given data (Not SCF)",
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
    args = parser.parse_args()

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

    main(**argdict)


if __name__ == "__main__":
    cli()
