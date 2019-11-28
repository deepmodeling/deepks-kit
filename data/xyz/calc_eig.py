import numpy as np
from scipy.spatial.distance import squareform, pdist


def load_coords(filename):
    return np.loadtxt(filename, skiprows=2, usecols=[1,2,3])


def cosine_switching(x, lower=1.9, upper=2.0, threshold=1e-5):
    zx = x < threshold
    lx = x < lower
    ux = x > upper
    mx = (~lx) & (~ux)
    res = np.zeros_like(x)
    res[~zx & lx] = 1
    res[mx] = 0.5*np.cos(np.pi * (x[mx]-lower) / (upper-lower)) + 0.5
    return res


def calc_weight(coords, lower=1.9, upper=2.0):
    natom = coords.shape[0]
    pair_dist = squareform(pdist(coords))
    weight = cosine_switching(pair_dist, lower, upper).reshape(1, natom, natom)
    return weight


def split(ci, shell):
    sec = [1]*shell[0] + [3]*shell[1] + [5]*shell[2]
    assert np.sum(sec) == ci.shape[-1]
    ci_list = np.split(ci, np.cumsum(sec)[:-1], axis=-1)
    return ci_list


def calc_atom_eig(ci, shell=(12,12,12), frozen=0):
    ci_list = split(ci[:, frozen:], shell)
    dm_list = [np.einsum('niap,niaq->napq', _ci, _ci) for _ci in ci_list]
    eig_list = [np.linalg.eigvalsh(dm) for dm in dm_list]
    eig = np.concatenate(eig_list, -1)
    return eig


def calc_atom_ener_eig(ci, ei, kernel=None, shell=(12,12,12), frozen=0):
    if kernel is not None:
        ei = kernel(ei)
    ci_list = split(ci[:, frozen:], shell)
    dm_list = [np.einsum('niap,niaq,ni->napq', _ci, _ci, ei[:, frozen:]) for _ci in ci_list]
    eig_list = [np.linalg.eigvalsh(dm) for dm in dm_list]
    eig = np.concatenate(eig_list, -1)
    return eig


def calc_neighbor_eig(ci, weight=None, shell=(12,12,12), frozen=0):
    ci_list = split(ci[:, frozen:], shell)
    dm_list = [np.einsum('niap,nibq->nabpq', _ci, _ci) for _ci in ci_list]
    if weight is not None:
        dm_list = [np.einsum('nabpq,nab->nabpq', _dm, weight) for _dm in dm_list]
    eig_list = [np.linalg.eigvalsh(0.5*(_dm.sum(1) + _dm.sum(2))) for _dm in dm_list]
    eig = np.concatenate(eig_list, -1)
    return eig


def calc_eig(name, ci, ei=None, xyz_file=None, shell=(12,12,12)):
    if name == 'dm_eig':
        return calc_atom_eig(ci, shell=shell)
    if name == 'od_eig':
        assert xyz_file is not None
        return calc_neighbor_eig(ci, calc_weight(load_coords(xyz_file)), shell=shell)
    if name == 'se_eig':
        assert ei is not None
        return calc_atom_ener_eig(ci, ei, kernel=None, shell=shell)
    if name == 'fe_eig':
        assert ei is not None
        return calc_atom_ener_eig(ci, ei, kernel=np.exp, shell=shell)

    raise ValueError(f'unsupport name: {name}')