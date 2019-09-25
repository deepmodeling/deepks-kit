import argparse, os
import numpy as np
import torch
import torch.nn as nn
from model import QCNet
from reader import GroupReader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model, g_reader, prefix="test"):
    loss_fn=nn.MSELoss()
    for i in range(g_reader.nsystems):
        sample = g_reader.sample_all(i)
        nframes = sample[0].shape[0]
        label, *data = [torch.from_numpy(d).to(DEVICE) for d in sample]
        pred = model(*data)
        error = torch.sqrt(loss_fn(pred, label))

        error_np = error.item() / g_reader.ec_scale
        label_np = label.cpu().numpy().reshape(nframes, -1).sum(axis=1) / g_reader.ec_scale
        pred_np = pred.detach().cpu().numpy().reshape(nframes, -1).sum(axis=1) / g_reader.ec_scale
        dump_res = np.stack([label_np, pred_np], axis=1)
        error_sys = np.sqrt(np.mean((label_np - pred_np)**2))
        
        header = f"{g_reader.path_list[i]}\nmean l2 error: {error_np}\nsystem l2 error: {error_sys}\nreal_ene  pred_ene"
        filename = f"{prefix}.{i}.out"
        np.savetxt(filename, dump_res, header=header)
        print(f"system {i} finished")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-file", default='model.pth', type=str,
                        help="the dumped model file to test")
    parser.add_argument("-d", "--data-path", default='data', type=str, nargs = '+',
                        help="the path to data .raw files for test")
    parser.add_argument("-o", "--output-prefix", default = 'test', type = str,
                        help=r"the prefix of output file, would wite into file %%prefix.%%sysidx.out")
    parser.add_argument("-s", "--ec-scale", default = 1.0, type = float,
                        help="the scale that data loader multiplies to correlation energy")
    args = parser.parse_args()

    model = QCNet.load(args.model_file).double().to(DEVICE)
    g_reader = GroupReader(args.data_path, 1, args.ec_scale)
    test(model, g_reader, prefix=args.output_prefix)


if __name__ == "__main__":
    main()