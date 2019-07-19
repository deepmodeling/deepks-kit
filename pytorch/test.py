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
        label, *data = [torch.from_numpy(d).to(DEVICE) for d in sample]
        pred = model(*data)
        error = torch.sqrt(loss_fn(pred, label))

        error_np = error.item()
        label_np = label.cpu().numpy().reshape(-1, 1)
        pred_np = pred.detach().cpu().numpy().reshape(-1, 1)
        dump_res = np.concatenate([label_np, pred_np], axis=1)
        
        header = f"{g_reader.path_list[i]}\nmean l2 error: {error_np}\nreal_ene  pred_ene"
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
                        help=r"the prefix of output file, would wite into file %prefix.%sysidx.out")
    args = parser.parse_args()

    model = QCNet.load(args.model_file).double().to(DEVICE)
    g_reader = GroupReader(args.data_path, 1)
    test(model, g_reader, prefix=args.output_prefix)


if __name__ == "__main__":
    main()