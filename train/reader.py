import os,time,sys
import numpy as np


class Reader(object):
    def __init__(self, data_path, batch_size, e_name="e_cc", d_name="dm_eig"):
        # copy from config
        self.data_path = data_path
        self.batch_size = batch_size   
        self.e_name = e_name
        self.d_name = d_name if isinstance(d_name, (list, tuple)) else [d_name]

    def prepare(self):
        self.index_count_all = 0
        sys_meta = np.loadtxt(os.path.join(self.data_path,'system.raw'), dtype = int).reshape([-1])
        self.meta = sys_meta
        self.natm = self.meta[0]
        self.nao = self.meta[1]
        self.nocc = self.meta[2]
        self.nvir = self.meta[3]
        # self.nproj = self.meta[4]
        self.data_ec = np.load(os.path.join(self.data_path,f'{self.e_name}.npy')).reshape([-1, 1])
        self.nframes = self.data_ec.shape[0]
        self.data_dm = np.concatenate(
            [np.load(os.path.join(self.data_path,f'{dn}.npy')).reshape([self.nframes, self.natm, -1])
                for dn in self.d_name], 
            axis=-1)
        self.nproj = self.data_dm.shape[-1]
        # print(np.shape(self.inputs_train))
        if self.nframes < self.batch_size:
            self.batch_size = self.nframes
            print('#', self.data_path, f"reset batch size to {self.batch_size}")
    
    def sample_train(self):
        if self.nframes == self.batch_size == 1:
            return self.data_ec, self.data_dm
        self.index_count_all += self.batch_size
        if self.index_count_all > self.nframes:
            # shuffle the data
            self.index_count_all = self.batch_size
            ind = np.random.choice(self.nframes, self.nframes, replace=False)
            self.data_ec = self.data_ec[ind]
            self.data_dm = self.data_dm[ind]
        ind = np.arange(self.index_count_all - self.batch_size, self.index_count_all)
        return \
            self.data_ec[ind], \
            self.data_dm[ind]

    def sample_all(self) :
        return \
            self.data_ec, \
            self.data_dm

    def get_train_size(self) :
        return self.nframes

    def get_batch_size(self) :
        return self.batch_size

    def get_data(self):
        return self.data_dm

    def get_nframes(self) :
        return self.nframes


class GroupReader(object) :
    def __init__ (self, path_list, batch_size=1, group_batch=1, e_name="e_cc", d_name="dm_eig") :
        self.path_list = path_list
        self.batch_size = batch_size
        self.nsystems = len(self.path_list)
        # init system readers
        self.readers = []
        for ii in self.path_list :
            self.readers.append(Reader(ii, batch_size, e_name=e_name, d_name=d_name))
        # prepare all systems
        for ii in self.readers:
            ii.prepare()
        # probability of each system
        self.nframes = []
        for ii in self.readers :
            self.nframes.append(ii.get_nframes())
        self.sys_prob = [float(ii) for ii in self.nframes] / np.sum(self.nframes)
        
        self.group_batch = max(group_batch, 1)
        if self.group_batch > 1:
            self.group_dict = {}
            self.group_index = {}
            for idx, r in enumerate(self.readers):
                if r.natm in self.group_dict:
                    self.group_dict[r.natm].append(r)
                    # self.group_index[r.natm].append(idx)
                else:
                    self.group_dict[r.natm] = [r]
                    # self.group_index[r.natm] = [idx]
            self.group_prob = {n: sum(r.nframes for r in r_list) / sum(self.nframes)
                                for n, r_list in self.group_dict.items()}
            self.batch_prob_raw = {n: [r.nframes / r.batch_size for r in r_list] 
                                for n, r_list in self.group_dict.items()}
            self.batch_prob = {n: p / np.sum(p) for n, p in self.batch_prob_raw.items()}

        self._sample_used = 0


    def __iter__(self):
        return self

    def __next__(self):
        if self._sample_used > self.get_train_size():
            self._sample_used = 0
            raise StopIteration
        sample = self.sample_train() if self.group_batch == 1 else self.sample_train_group()
        self._sample_used += sample[0].shape[0]
        return sample

    def sample_idx(self) :
        return np.random.choice(np.arange(self.nsystems), p=self.sys_prob)
        
    def sample_train(self, idx=None) :
        if idx is None:
            idx = self.sample_idx()
        return \
            self.readers[idx].sample_train()

    def sample_train_group(self):
        cnatm = np.random.choice(list(self.group_prob.keys()), p=list(self.group_prob.values()))
        cgrp = self.group_dict[cnatm]
        csys = np.random.choice(cgrp, self.group_batch, p=self.batch_prob[cnatm])
        all_sample = list(zip(*[s.sample_train() for s in csys]))
        return [np.concatenate(d, axis=0) for d in all_sample]

    def sample_all(self, idx=None) :
        if idx is None:
            idx = self.sample_idx()
        return \
            self.readers[idx].sample_all()
    
    def sample_all_batch(self, idx=None):
        if idx is not None:
            all_data = self.sample_all(idx)
            n_split = all_data[0].shape[0] // self.batch_size
            yield from zip(*[np.array_split(all_data[i], n_split, axis=0) for i in range(len(all_data))])
        else:
            for i in range(self.nsystems):
                yield from self.sample_all_batch(i)

    def get_train_size(self) :
        return np.sum(self.nframes)

    def get_batch_size(self) :
        return self.batch_size

    def compute_data_stat(self):
        if not (hasattr(self, 'all_mean') and hasattr(self, 'all_std')):
            all_dm = np.concatenate([r.data_dm.reshape(-1,r.nproj) for r in self.readers])
            self.all_mean, self.all_std = all_dm.mean(0), all_dm.std(0)
        return self.all_mean, self.all_std

    def compute_prefitting(self, shift=None, scale=None, ridge_alpha=0):
        all_mean, all_std = self.compute_data_stat()
        if shift is None:
            shift = all_mean
        if scale is None:
            scale = all_std
        all_sdm = np.concatenate([((r.data_dm - shift) / scale).sum(1) for r in self.readers])
        all_natm = np.concatenate([[float(r.data_dm.shape[1])]*r.data_dm.shape[0] for r in self.readers])
        all_x = np.concatenate([all_sdm, all_natm.reshape(-1,1)], -1)
        all_y = np.concatenate([r.data_ec for r in self.readers])
        
        from sklearn.linear_model import Ridge 
        reg = Ridge(alpha=ridge_alpha, fit_intercept=False, tol=1e-9)
        reg.fit(all_x, all_y)
        coef = reg.coef_.reshape(-1)
        # coef, _, _, _ = np.linalg.lstsq(all_x, all_y, None)
        return coef[:-1], coef[-1]
