import os,time,sys
import numpy as np

class Reader(object):
    def __init__(self, data_path, batch_size, ec_scale=1.0):
        # copy from config
        self.data_path = data_path
        self.batch_size = batch_size   
        self.ec_scale = ec_scale

    def prepare(self):
        self.index_count_all = 0
        sys_meta = np.loadtxt(os.path.join(self.data_path,'system.raw'), dtype = int).reshape([-1])
        self.meta = sys_meta
        self.natm = self.meta[0]
        self.nao = self.meta[1]
        self.nocc = self.meta[2]
        self.nvir = self.meta[3]
        self.nproj = self.meta[4]
        self.tr_data_ec = np.load(os.path.join(self.data_path,'e_mp2.npy')).reshape([-1])
        nframes = self.tr_data_ec.shape[0]
        self.tr_data_ec_ij = np.load(os.path.join(self.data_path, 'ec_ij.npy')).reshape([nframes, self.nocc, self.nocc])
        self.tr_data_ec_i = self.ec_scale * self.tr_data_ec_ij.sum(axis=1)
        
        self.tr_data_mo_occ = np.load(os.path.join(self.data_path,'coeff_occ.npy')).reshape([nframes, self.nocc, self.natm, self.nproj, 1])
        self.tr_data_mo_vir = np.load(os.path.join(self.data_path,'coeff_vir.npy')).reshape([nframes, self.nvir, self.natm, self.nproj, 1])
        self.tr_data_e_occ = np.load(os.path.join(self.data_path,'ener_occ.npy')).reshape([nframes, self.nocc])
        self.tr_data_e_vir = np.load(os.path.join(self.data_path,'ener_vir.npy')).reshape([nframes, self.nvir])
        self.train_size_all = nframes
        # print(np.shape(self.inputs_train))
        if self.train_size_all < self.batch_size:
            self.batch_size = self.train_size_all
            print('#', self.data_path, f"reset batch size to {self.batch_size}")
    
    def _sample_train_all(self):
        self.index_count_all += self.batch_size
        if self.index_count_all > self.train_size_all:
            # shuffle the data
            self.index_count_all = self.batch_size
            ind = np.random.choice(self.train_size_all, self.train_size_all, replace=False)
            self.tr_data_ec = self.tr_data_ec[ind]
            self.tr_data_mo_occ = self.tr_data_mo_occ[ind]
            self.tr_data_mo_vir = self.tr_data_mo_vir[ind]
            self.tr_data_e_occ = self.tr_data_e_occ[ind]
            self.tr_data_e_vir = self.tr_data_e_vir[ind]
        ind = np.arange(self.index_count_all - self.batch_size, self.index_count_all)
        return \
            self.tr_data_ec[ind], \
            self.tr_data_mo_occ[ind], \
            self.tr_data_mo_vir[ind], \
            self.tr_data_e_occ[ind], \
            self.tr_data_e_vir[ind]

    def sample_train(self) :
        return self._sample_train_all()

    def sample_all(self) :
        return \
            self.tr_data_ec, \
            self.tr_data_mo_occ, \
            self.tr_data_mo_vir, \
            self.tr_data_e_occ, \
            self.tr_data_e_vir

    def get_train_size(self) :
        return self.train_size_all

    def get_batch_size(self) :
        return self.batch_size

    def get_data(self):
        return self.tr_data_mo_occ, self.tr_data_mo_vir, self.tr_data_e_occ, self.tr_data_e_vir

    # def get_meta(self): 
    #     return self.natm, self.nao, self.nocc, self.nvir, self.nproj
    def get_meta(self) :
        return self.meta

    def get_nframes(self) :
        return self.tr_data_ec.shape[0]


class GroupReader(object) :
    def __init__ (self, path_list, batch_size, ec_scale=1.0) :
        self.path_list = path_list
        self.batch_size = batch_size
        self.ec_scale = ec_scale
        self.nsystems = len(self.path_list)
        # init system readers
        self.readers = []
        for ii in self.path_list :
            self.readers.append(Reader(ii, batch_size, ec_scale))
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

    def compute_ener_stat(self) :
        all_e = np.array([])
        for ii in self.readers :
            _, _, e_occ, e_vir = ii.get_data()
            e_occ = e_occ.reshape([-1])
            e_vir = e_vir.reshape([-1])
            all_e = np.concatenate((all_e, e_occ, e_vir))
        return np.average(all_e), np.std(all_e)

    def compute_coeff_stat(self) :
        avg_list = []
        var_list = []
        for ii in self.readers :
            mo_occ, mo_vir, _, _ = ii.get_data()
            mo_all = np.concatenate((mo_occ, mo_vir), axis=1)
            assert len(mo_all.shape) == 5
            avg_list.append(np.mean(mo_all, axis=(0,1,2,3)))
            var_list.append(np.var(mo_all, axis=(0,1,2,3)))
        return \
            np.average(avg_list, weights=self.nframes, axis=0), \
            np.sqrt(np.average(var_list, weights=self.nframes, axis=0))