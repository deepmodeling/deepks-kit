import os,time,sys
import numpy as np

class Reader(object):
    def __init__(self, data_path, batch_size, seed = None):
        # copy from config
        self.data_path = data_path
        self.batch_size = batch_size   
        np.random.seed(seed)

    def prepare(self):
        self.index_count_all = 0
        sys_meta = np.loadtxt(os.path.join(self.data_path,'system.raw'), dtype = int).reshape([-1])
        self.meta = sys_meta
        self.natm = self.meta[0]
        self.nao = self.meta[1]
        self.nocc = self.meta[2]
        self.nvir = self.meta[3]
        self.nproj = self.meta[4]
        self.tr_data_emp2 = np.loadtxt(os.path.join(self.data_path,'e_mp2.raw')).reshape([-1])
        nframes = self.tr_data_emp2.shape[0]
        self.tr_data_dist = np.loadtxt(os.path.join(self.data_path,'dist.raw')).reshape([-1])
        self.tr_data_dist = np.ones(self.tr_data_dist.shape)
        assert(nframes == self.tr_data_dist.shape[0])
        self.tr_data_mo_occ = np.loadtxt(os.path.join(self.data_path,'coeff_occ.raw')).reshape([nframes, self.nocc, self.natm, self.nproj])
        self.tr_data_mo_vir = np.loadtxt(os.path.join(self.data_path,'coeff_vir.raw')).reshape([nframes, self.nvir, self.natm, self.nproj])
        self.tr_data_e_occ = np.loadtxt(os.path.join(self.data_path,'ener_occ.raw')).reshape([nframes, self.nocc])
        self.tr_data_e_vir = np.loadtxt(os.path.join(self.data_path,'ener_vir.raw')).reshape([nframes, self.nvir])
        self.train_size_all = nframes
        # print(np.shape(self.inputs_train))
    
    def _sample_train_all(self):
        self.index_count_all += self.batch_size
        if self.index_count_all > self.train_size_all:
            # shuffle the data
            self.index_count_all = self.batch_size
            ind = np.random.choice(self.train_size_all, self.train_size_all, replace=False)
            self.tr_data_emp2 = self.tr_data_emp2[ind]
            self.tr_data_dist = self.tr_data_dist[ind]
            self.tr_data_mo_occ = self.tr_data_mo_occ[ind]
            self.tr_data_mo_vir = self.tr_data_mo_vir[ind]
            self.tr_data_e_occ = self.tr_data_e_occ[ind]
            self.tr_data_e_vir = self.tr_data_e_vir[ind]
        ind = np.arange(self.index_count_all - self.batch_size, self.index_count_all)
        return \
            self.tr_data_emp2[ind], \
            1./self.tr_data_dist[ind], \
            self.tr_data_mo_occ[ind], \
            self.tr_data_mo_vir[ind], \
            self.tr_data_e_occ[ind], \
            self.tr_data_e_vir[ind]

    def sample_train(self) :
        return self._sample_train_all()

    def sample_all(self) :
        return \
            self.tr_data_emp2, \
            1./self.tr_data_dist, \
            self.tr_data_mo_occ[:], \
            self.tr_data_mo_vir[:], \
            self.tr_data_e_occ[:], \
            self.tr_data_e_vir[:]

    def get_train_size(self) :
        return self.train_size_all

    def get_batch_size(self) :
        return self.batch_size

    def get_data(self):
        return 1./self.tr_data_dist, self.tr_data_mo_occ, self.tr_data_mo_vir, self.tr_data_e_occ, self.tr_data_e_vir

    # def get_meta(self): 
    #     return self.natm, self.nao, self.nocc, self.nvir, self.nproj
    def get_meta(self) :
        return self.meta

    def get_nframes(self) :
        return self.tr_data_emp2.shape[0]


class GroupReader(object) :
    def __init__ (self,path_list, batch_size, seed = None) :
        self.path_list = path_list
        self.batch_size = batch_size
        self.nsystems = len(self.path_list)
        # init system readers
        self.readers = []
        for ii in self.path_list :
            self.readers.append(Reader(ii, batch_size, seed))
        # prepare all systems
        for ii in self.readers:
            ii.prepare()
        # probability of each system
        self.nframes = []
        for ii in self.readers :
            self.nframes.append(ii.get_nframes())
        self.sys_prob = [float(ii) for ii in self.nframes] / np.sum(self.nframes)

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
            tmp0, tmp1, tmp2, e_occ, e_vir = ii.get_data()
            e_occ = e_occ.reshape([-1])
            e_vir = e_vir.reshape([-1])
            all_e = np.concatenate((all_e, e_occ, e_vir))
        return np.average(all_e), np.std(all_e)