import os
import numpy as np
import torch
import torch.nn as nn
try:
    import deepqc
except ImportError as e:
    import sys
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepqc.train.model import CorrNet
from deepqc.train.reader import GroupReader
from deepqc.utils import load_yaml, load_dirs, check_list


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
    data_paths = load_dirs(data_paths)
    g_reader = GroupReader(data_paths, e_name=e_name, d_name=d_name)
    model_file = check_list(model_file)
    for f in model_file:
        print(f)
        p = os.path.dirname(f)
        model = CorrNet.load(f).double().to(DEVICE)
        dump = os.path.join(p, output_prefix)
        dir_name = os.path.dirname(dump)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        test(model, g_reader, dump_prefix=dump, group=group)


if __name__ == "__main__":
    from deepqc.main import test_cli as cli
    cli()
