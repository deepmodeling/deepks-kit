import os,time,sys
import numpy as np

class Reader(object):
    def __init__(self, data_path, batch_size):
        # copy from config
        self.data_path = data_path
        self.batch_size = batch_size   

    def prepare(self):
        self.index_count_all = 0
        sys_meta = np.loadtxt(os.path.join(self.data_path,'system.raw'), dtype = int).reshape([-1])
        self.meta = sys_meta
        self.natm = self.meta[0]
        self.nao = self.meta[1]
        self.nocc = self.meta[2]
        self.nvir = self.meta[3]
        self.nproj = self.meta[4]
        self.data_ec = np.load(os.path.join(self.data_path,'e_mp2.npy')).reshape([-1, 1])
        self.nframes = self.data_ec.shape[0]
        self.data_eig_occ = np.load(os.path.join(self.data_path,'eig_occ.npy')).reshape([self.nframes, -1, self.nocc])
        self.nshell = self.data_eig_occ.shape[1]
        # print(np.shape(self.inputs_train))
        if self.nframes < self.batch_size:
            self.batch_size = self.nframes
            print('#', self.data_path, f"reset batch size to {self.batch_size}")
    
    def sample_train(self):
        if self.nframes == self.batch_size == 1:
            return self.data_ec, self.data_eig_occ
        self.index_count_all += self.batch_size
        if self.index_count_all > self.nframes:
            # shuffle the data
            self.index_count_all = self.batch_size
            ind = np.random.choice(self.nframes, self.nframes, replace=False)
            self.data_ec = self.data_ec[ind]
            self.data_eig_occ = self.data_eig_occ[ind]
        ind = np.arange(self.index_count_all - self.batch_size, self.index_count_all)
        return \
            self.data_ec[ind], \
            self.data_eig_occ[ind]

    def sample_all(self) :
        return \
            self.data_ec, \
            self.data_eig_occ

    def get_train_size(self) :
        return self.nframes

    def get_batch_size(self) :
        return self.batch_size

    def get_data(self):
        return self.data_eig_occ

    def get_meta(self) :
        return self.meta

    def get_nframes(self) :
        return self.nframes


class GroupReader(object) :
    def __init__ (self, path_list, batch_size) :
        self.path_list = path_list
        self.batch_size = batch_size
        self.nsystems = len(self.path_list)
        # init system readers
        self.readers = []
        for ii in self.path_list :
            self.readers.append(Reader(ii, batch_size))
        # prepare all systems
        for ii in self.readers:
            ii.prepare()
        # probability of each system
        self.nframes = []
        for ii in self.readers :
            self.nframes.append(ii.get_nframes())
        self.sys_prob = [float(ii) for ii in self.nframes] / np.sum(self.nframes)
        self._sample_used = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._sample_used > self.get_train_size():
            self._sample_used = 0
            raise StopIteration
        self._sample_used += self.batch_size
        return self.sample_train()

    def sample_idx(self) :
        return np.random.choice(np.arange(self.nsystems), p = self.sys_prob)
        
    def sample_meta(self, idx=None) :
        if idx is None:
            idx = self.sample_idx()
        return self.readers[idx].get_meta()

    def sample_train(self, idx=None) :
        if idx is None:
            idx = self.sample_idx()
        return \
            self.readers[idx].sample_train()

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

    def get_nvec_dof(self) :
        return self.readers[0].get_meta()[0] * self.readers[0].get_meta()[4]

    def get_nproj(self) :
        return self.readers[0].get_meta()[4]

    def get_nmeta(self) :
        return len(self.readers[0].get_meta())
