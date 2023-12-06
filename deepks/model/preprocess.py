
import numpy as np


def compute_data_stat(greader, symm_sections=None):

    all_dm = np.concatenate([r.data_dm.reshape(-1, r.ndesc) for r in greader.readers])
    if symm_sections is None:
        all_mean, all_std = all_dm.mean(0), all_dm.std(0)
    else:
        assert sum(symm_sections) == all_dm.shape[-1]
        dm_shells = np.split(all_dm, np.cumsum(symm_sections)[:-1], axis=-1)
        mean_shells = [d.mean().repeat(s) for d, s in zip(dm_shells, symm_sections)]
        std_shells = [d.std().repeat(s) for d, s in zip(dm_shells, symm_sections)]
        all_mean = np.concatenate(mean_shells, axis=-1)
        all_std = np.concatenate(std_shells, axis=-1)
    return all_mean, all_std


def compute_prefitting(greader, shift=None, scale=None, ridge_alpha=1e-8, symm_sections=None):
    if shift is None or scale is None:
        all_mean, all_std = compute_data_stat(greader, symm_sections=symm_sections)
        if shift is None:
            shift = all_mean
        if scale is None:
            scale = all_std
    all_sdm = np.concatenate([((r.data_dm - shift) / scale).sum(1) for r in greader.readers])
    all_natm = np.concatenate([[float(r.data_dm.shape[1])]*r.data_dm.shape[0] for r in greader.readers])
    if symm_sections is not None: # in this case ridge alpha cannot be 0
        assert sum(symm_sections) == all_sdm.shape[-1]
        sdm_shells = np.split(all_sdm, np.cumsum(symm_sections)[:-1], axis=-1)
        all_sdm = np.stack([d.sum(-1) for d in sdm_shells], axis=-1)
    # build feature matrix
    X = np.concatenate([all_sdm, all_natm.reshape(-1,1)], -1)
    y = np.concatenate([r.data_ec for r in greader.readers])
    I = np.identity(X.shape[1])
    I[-1,-1] = 0 # do not punish the bias term
    # solve ridge reg
    coef = np.linalg.solve(X.T @ X + ridge_alpha * I, X.T @ y).reshape(-1)
    weight, bias = coef[:-1], coef[-1]
    if symm_sections is not None:
        weight = np.concatenate([w.repeat(s) for w, s in zip(weight, symm_sections)], axis=-1)
    return weight, bias


def preprocess(model, g_reader,
               preshift=True, prescale=False, prescale_sqrt=False, prescale_clip=0,
               prefit=True, prefit_ridge=10, prefit_trainable=False):
    shift = model.input_shift.cpu().detach().numpy()
    scale = model.input_scale.cpu().detach().numpy()
    symm_sec = model.shell_sec # will be None if no embedding
    prefit_trainable = prefit_trainable and symm_sec is None # no embedding
    if preshift or prescale:
        davg, dstd = compute_data_stat(g_reader, symm_sec)
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
        weight, bias = compute_prefitting(g_reader,
                                          shift=shift, scale=scale,
                                          ridge_alpha=prefit_ridge, symm_sections=symm_sec)
        model.set_prefitting(weight, bias, trainable=prefit_trainable)
