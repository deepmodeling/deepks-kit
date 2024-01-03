import os
import numpy as np
import torch
import torch.nn as nn
try:
    import deepks
except ImportError as e:
    import sys
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepks.model.model import CorrNet
from deepks.model.model_enn import CorrNet as CorrNetEquiv
from deepks.model.reader import GroupReader
from deepks.utils import load_yaml, load_dirs, check_list


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, g_reader, dump_prefix="test", group=False):
    model.eval()
    loss_fn=nn.MSELoss()
    label_list = []
    pred_list = []

    for i in range(g_reader.nsystems):
        sample = g_reader.sample_all(i)
        nframes = sample["lb_e"].shape[0]
        sample = {k: v.to(DEVICE, non_blocking=True) for k, v in sample.items()}
        label, data = sample["lb_e"], sample["desc"]
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


def infer_dname_from_model(model_file):

    checkpoint = torch.load(model_file, map_location="cpu")
    d_name = "dm_eig"
    if "model_type" in checkpoint.keys() and checkpoint["model_type"] == "equivariant":
        d_name = "dm_flat"
    return [d_name]


def main(data_paths, model_file="model.pth", 
         output_prefix='test', group=False,
         e_name='l_e_delta', d_name=None):
    data_paths = load_dirs(data_paths)
    f = model_file
    if isinstance(model_file, list):
        f = model_file[0]
    d_name = infer_dname_from_model(f)
    Net = CorrNetEquiv if 'dm_flat' in d_name else CorrNet
    if len(d_name) == 1:
        d_name = d_name[0]
    g_reader = GroupReader(data_paths, e_name=e_name, d_name=d_name, 
                           conv_filter=False, extra_label=True)
    model_file = check_list(model_file)
    for f in model_file:
        print(f)
        p = os.path.dirname(f)
        model = Net.load(f).double().to(DEVICE)
        dump = os.path.join(p, output_prefix)
        dir_name = os.path.dirname(dump)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if isinstance(model, CorrNet) and model.elem_table is not None:
            elist, econst = model.elem_table
            g_reader.collect_elems(elist)
            g_reader.subtract_elem_const(econst)
        test(model, g_reader, dump_prefix=dump, group=group)
        g_reader.revert_elem_const()


if __name__ == "__main__":
    from deepks.main import test_cli as cli
    cli()
