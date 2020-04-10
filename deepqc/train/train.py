import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_sample(model, sample, loss_fn=nn.MSELoss()):
    label, *data = [torch.from_numpy(d).to(DEVICE) for d in sample]
    pred = model(*data)
    loss = loss_fn(pred, label)
    return loss


def preprocess(model, g_reader, 
                preshift=False, prescale=False, prescale_sqrt=False, prescale_clip=0,
                prefit=True, prefit_ridge=0, prefit_trainable=False):
    shift = model.input_shift.cpu().detach().numpy()
    scale = model.input_scale.cpu().detach().numpy()
    if preshift or prescale:
        davg, dstd = g_reader.compute_data_stat()
        if preshift: 
            shift = davg
        if prescale: 
            scale = dstd
            if prescale_sqrt: 
                scale = np.sqrt(scale)
            if prescale_clip: 
                scale = scale.clip(prescale_clip)
        model.set_normalization(shift, scale)
    if prefit:
        weight, bias = g_reader.compute_prefitting(shift=shift, scale=scale, ridge_alpha=prefit_ridge)
        model.set_prefitting(weight, bias, trainable=prefit_trainable)


def train(model, g_reader, n_epoch, 
            test_reader=None,
            start_lr=0.01, decay_steps=100, decay_rate=0.96, weight_decay=0.0,
            display_epoch=100, ckpt_file=None):
    if test_reader is None:
        test_reader = g_reader
    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, decay_steps, decay_rate)
    loss_fn = nn.MSELoss()

    print("# working on device:", DEVICE)
    print("# epoch      trn_err   tst_err        lr  trn_time  tst_time ")
    tic = time()
    trn_loss = np.mean([eval_sample(model, batch).item() for batch in g_reader.sample_all_batch()])
    tst_loss = np.mean([eval_sample(model, batch).item() for batch in test_reader.sample_all_batch()])
    tst_time = time() - tic
    print(f"  {0:<8d}  {np.sqrt(trn_loss):>.2e}  {np.sqrt(tst_loss):>.2e}  {start_lr:>.2e}  {0:>8.2f}  {tst_time:>8.2f}")

    for epoch in range(1, n_epoch+1):
        tic = time()
        loss_list = []
        for sample in g_reader:
            label, *data = [torch.from_numpy(d).to(DEVICE) for d in sample]
            optimizer.zero_grad()
            pred = model(*data)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        scheduler.step()

        if epoch % display_epoch == 0:
            trn_loss = np.mean(loss_list)
            trn_time = time() - tic
            tic = time()
            tst_loss = np.mean([eval_sample(model, batch).item() for batch in test_reader.sample_all_batch()])
            tst_time = time() - tic
            print(f"  {epoch:<8d}  {np.sqrt(trn_loss):>.2e}  {np.sqrt(tst_loss):>.2e}  {scheduler.get_lr()[0]:>.2e}  {trn_time:>8.2f}  {tst_time:8.2f}")
            if ckpt_file:
                model.save(ckpt_file)
    
    


# if __name__ == "__main__":
#     from model import QCNet 
#     from reader import GroupReader
#     g_reader = GroupReader(["/data1/yixiaoc/work/deep.qc/data/wanghan/data_B_B"], 32)
#     model = QCNet([40,40,40], [40,40,40], 
#                     e_stat=g_reader.compute_ener_stat(), 
#                     use_resnet=True).double().to(DEVICE)
#     train(model, g_reader, 1000)
