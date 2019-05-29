#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,time,sys
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

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
        self.tr_data_mo_occ = np.loadtxt(os.path.join(self.data_path,'coeff_occ.raw')).reshape([nframes,self.nocc*self.natm*self.nproj])
        self.tr_data_mo_vir = np.loadtxt(os.path.join(self.data_path,'coeff_vir.raw')).reshape([nframes,self.nvir*self.natm*self.nproj])
        self.tr_data_e_occ = np.loadtxt(os.path.join(self.data_path,'ener_occ.raw')).reshape([nframes,self.nocc])
        self.tr_data_e_vir = np.loadtxt(os.path.join(self.data_path,'ener_vir.raw')).reshape([nframes,self.nvir])
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
            self.tr_data_mo_occ = self.tr_data_mo_occ[ind,:]
            self.tr_data_mo_vir = self.tr_data_mo_vir[ind,:]
            self.tr_data_e_occ = self.tr_data_e_occ[ind,:]
            self.tr_data_e_vir = self.tr_data_e_vir[ind,:]
        ind = np.arange(self.index_count_all - self.batch_size, self.index_count_all)
        return \
            self.tr_data_emp2[ind], \
            1./self.tr_data_dist[ind], \
            self.tr_data_mo_occ[ind, :].reshape([-1]), \
            self.tr_data_mo_vir[ind, :].reshape([-1]), \
            self.tr_data_e_occ[ind, :].reshape([-1]), \
            self.tr_data_e_vir[ind, :].reshape([-1])

    def sample_train(self) :
        return self._sample_train_all()

    def sample_all(self) :
        return \
            self.tr_data_emp2, \
            1./self.tr_data_dist, \
            self.tr_data_mo_occ[:, :].reshape([-1]), \
            self.tr_data_mo_vir[:, :].reshape([-1]), \
            self.tr_data_e_occ[:, :].reshape([-1]), \
            self.tr_data_e_vir[:, :].reshape([-1])

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
        
    def sample_meta(self, idx) :
        return self.readers[idx].get_meta()

    def sample_train(self, idx) :
        return \
            self.readers[idx].sample_train()

    def sample_all(self, idx) :
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
    

class Model(object):
    def __init__(self, config, sess):
        self.sess = sess
        # copy from config
        self.data_path = config.data_path
        self.n_neuron = config.n_neuron
        self.n_displayepoch = config.n_displayepoch
        self.starter_learning_rate = config.starter_learning_rate
        self.decay_steps = config.decay_steps
        self.decay_rate = config.decay_rate
        self.display_in_training = config.display_in_training
        self.resnet = config.resnet
        self.with_ener = config.with_ener
        self.reg_weight = config.reg_weight

    def test_error (self, t_meta, t_emp2, t_dist, t_mo_occ, t_mo_vir, t_e_occ, t_e_vir) :
        ret = self.sess.run([self.l2_loss, self.emp2], 
                            feed_dict={self.emp2_ref: t_emp2,
                                       self.mo_occ: t_mo_occ,
                                       self.mo_vir: t_mo_vir,
                                       self.e_occ: t_e_occ,
                                       self.e_vir: t_e_vir,
                                       self.meta: t_meta,
                                       self.is_training: False})
        np.savetxt('tmp.out', np.concatenate((t_emp2, ret[1])).reshape(2,-1).T)
        error = np.sqrt(ret[0])
        # print(ret[1], t_emp2)
        return error


    def train(self, g_reader, num_epoch, seed = None):

        self.nvec_dof = g_reader.get_nvec_dof()
        nmeta = g_reader.get_nmeta()
        self.nproj = g_reader.get_nproj()

        # placeholders
        batch_size = g_reader.get_batch_size()
        self.mo_occ     = tf.placeholder(tf.float64, [None], name='input_mo_occ')
        self.mo_vir     = tf.placeholder(tf.float64, [None], name='input_mo_vir')
        self.e_occ      = tf.placeholder(tf.float64, [None], name='input_e_occ')
        self.e_vir      = tf.placeholder(tf.float64, [None], name='input_e_vir')
        self.emp2_ref   = tf.placeholder(tf.float64, [None], name='input_emp2')
        self.meta       = tf.placeholder(tf.int32, [nmeta], name='input_meta')
        self.is_training= tf.placeholder(tf.bool)

        # compute statistic 
        # data_stat = self.compute_statistic(reader)
        data_stat = g_reader.compute_ener_stat()

        # build dnn 
        self.emp2, self.w_l2 \
            = self.build_dnn(self.mo_occ, 
                             self.mo_vir, 
                             self.e_occ, 
                             self.e_vir, 
                             self.meta,
                             suffix = 'test',
                             stat = data_stat, 
                             reuse = False, 
                             seed = seed)
        self.l2_loss \
            = self.build_loss(self.emp2_ref, self.emp2, self.w_l2, suffix = 'test') 

        # learning rate
        self._extra_train_ops = []
        self.global_step \
            = tf.get_variable('global_step', 
                              [],
                              initializer = tf.constant_initializer(1),
                              trainable = False, 
                              dtype = tf.int32)
        self.global_epoch = self.global_step * g_reader.get_batch_size() // g_reader.get_train_size()
        self.learning_rate \
            = tf.train.exponential_decay(self.starter_learning_rate, 
                                         self.global_epoch,
                                         self.decay_steps,
                                         self.decay_rate, 
                                         staircase=True)

        # train operations
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.l2_loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=self.global_step, name='train_step')
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        # saver
        saver = tf.train.Saver()

        # parameter initialization
        sample_used = 0
        epoch_used = 0
        self.sess.run(tf.global_variables_initializer())
        print('# start training from scratch')
        start_time = time.time()

        # test initial error
        s_idx = g_reader.sample_idx()
        t_meta = g_reader.sample_meta(s_idx)
        ta_meta = g_reader.sample_meta(s_idx)
        t_emp2, t_dist, t_mo_occ, t_mo_vir, t_e_occ, t_e_vir\
            = g_reader.sample_train(s_idx)
        ta_emp2, ta_dist, ta_mo_occ, ta_mo_vir, ta_e_occ, ta_e_vir\
            = g_reader.sample_all(s_idx)
        error = self.test_error(t_meta, t_emp2, t_dist, t_mo_occ, t_mo_vir, t_e_occ, t_e_vir)
        error_a = self.test_error(ta_meta, ta_emp2, ta_dist, ta_mo_occ, ta_mo_vir, ta_e_occ, ta_e_vir)
        current_lr = self.sess.run(tf.to_double(self.learning_rate))
        if self.display_in_training:
            print ("epoch: %8u  ab_err: %.2e  ab_err_all: %.2e  lr: %.2e"
                   % (epoch_used, error, error_a, current_lr))

        # training
        train_time = 0
        while epoch_used < num_epoch:
            # exec training op
            tic = time.time()
            s_idx = g_reader.sample_idx()
            d_meta = g_reader.sample_meta(s_idx)
            d_emp2, d_dist, d_mo_occ, d_mo_vir, d_e_occ, d_e_vir = g_reader.sample_train(s_idx)
            self.sess.run([self.train_op], 
                          feed_dict={self.emp2_ref: d_emp2,
                                     self.mo_occ: d_mo_occ,
                                     self.mo_vir: d_mo_vir,
                                     self.e_occ: d_e_occ,
                                     self.e_vir: d_e_vir,
                                     self.meta : d_meta,
                                     self.is_training: True})
            train_time += time.time() - tic
            sample_used += g_reader.get_batch_size()
            # update number of epochs used, test and display if required
            if (sample_used // g_reader.get_train_size()) > epoch_used:
                epoch_used = sample_used // g_reader.get_train_size()
                if epoch_used % self.n_displayepoch == 0:
                    save_path = saver.save(self.sess, os.getcwd() + "/" + "model.ckpt")
                    tic = time.time()
                    error = self.test_error(d_meta, d_emp2, d_dist, d_mo_occ, d_mo_vir, d_e_occ, d_e_vir)
                    ta_emp2, ta_dist, ta_mo_occ, ta_mo_vir, ta_e_occ, ta_e_vir = g_reader.sample_all(s_idx)
                    error_a = self.test_error(d_meta, ta_emp2, ta_dist, ta_mo_occ, ta_mo_vir, ta_e_occ, ta_e_vir)
                    test_time = time.time() - tic
                    current_lr = self.sess.run(tf.to_double(self.learning_rate))
                    if self.display_in_training:
                        print ("epoch: %8u  ab_err: %.2e  ab_err_all: %.2e  lr: %.2e  trn_time %4.1f  tst_time %4.2f"
                               % (epoch_used, error, error_a, current_lr, train_time, test_time))                        
                        sys.stdout.flush()
                    train_time = 0

        # finalize
        end_time = time.time()
        print ("running time: %.3f s" % (end_time-start_time))


    def compute_one_statistic(self, 
                              mo_occ) :
        da_mo_occ = np.average (mo_occ, axis = 0)
        ds_mo_occ = np.std     (mo_occ, axis = 0)
        for ii in range(ds_mo_occ.size) :
            if np.abs(ds_mo_occ[ii]) < 1 :
                ds_mo_occ[ii] = 1
            else :
                ds_mo_occ[ii] = 1./ds_mo_occ[ii]
        return da_mo_occ, ds_mo_occ
        
    def compute_statistic(self, 
                          reader) :
        dist, mo_occ, mo_vir, e_occ, e_vir = reader.get_data()
        da_mo_occ, ds_mo_occ = self.compute_one_statistic(mo_occ)
        da_mo_vir, ds_mo_vir = self.compute_one_statistic(mo_vir)
        da_e_occ, ds_e_occ = self.compute_one_statistic(e_occ)
        da_e_vir, ds_e_vir = self.compute_one_statistic(e_vir)
        return \
            [da_mo_occ, ds_mo_occ, da_mo_vir, ds_mo_vir, \
             da_e_occ, ds_e_occ, da_e_vir, ds_e_vir]


    def build_loss(self, 
                   ener_ref,
                   ener, 
                   w_l2,
                   suffix = None) :
        ee_diff = ener_ref - ener
        ee_diff_norm = ee_diff * ee_diff
        l2_loss = tf.reduce_mean(tf.square(ener_ref - ener), 
                                 name='l2_loss_' + suffix)
        return l2_loss + self.reg_weight * w_l2

    
    def scale_var(self, 
                  mo_occ, 
                  shift,
                  scale, 
                  name, 
                  reuse = None) :
        assert(shift.size == scale.size)
        with tf.variable_scope(name, reuse=reuse):
            t_mo_occ_a\
                = tf.get_variable('avg', 
                                  [shift.size],
                                  dtype = tf.float64,
                                  trainable = False,
                                  initializer = tf.constant_initializer(shift))
            t_mo_occ_s\
                = tf.get_variable('std', 
                                  [scale.size],
                                  dtype = tf.float64,
                                  trainable = False,
                                  initializer = tf.constant_initializer(scale))
            mo_occ = (mo_occ - t_mo_occ_a) * t_mo_occ_s
        return mo_occ
    

    def build_dnn (self, 
                   mo_occ,
                   mo_vir,
                   e_occ,
                   e_vir,
                   meta,
                   stat = None,
                   suffix = None,
                   reuse = None, 
                   seed = None):                
        if stat is not None :           
            e_occ = self.scale_var(e_occ, stat[0], stat[1], 'e_stat', reuse = False)
            e_vir = self.scale_var(e_vir, stat[0], stat[1], 'e_stat', reuse = True)
        sys_ener,l2 = self.build_system_net(mo_occ, 
                                         mo_vir, 
                                         e_occ, 
                                         e_vir,
                                         meta,
                                         reuse = reuse,
                                         seed = seed)
        sys_ener = tf.identity(sys_ener, name = 'sys_ener_' + suffix)
        # l2 loss
        return sys_ener, l2

    def build_system_net(self, 
                         mo_occ, 
                         mo_vir, 
                         e_occ, 
                         e_vir,
                         meta,
                         reuse = None, 
                         seed = None):
        filter_neuron = [5,5,5]
        n_filter = filter_neuron[-1]
        # n_vec_dof = self.nvec_dof
        # nvec_dof = natm * nproj
        n_vec_dof = meta[0] * meta[4]
        # nframes x nocc -> (nframes x nocc) x M
        e_occ, weight_l2_occ = self.ds_layer(e_occ, filter_neuron, name = 'occ', reuse = reuse, seed = seed)
        # nframe x nocc x M
        e_occ = tf.reshape(e_occ, [-1, meta[2], n_filter])
        # nframe x nocc x nvec_dof
        mo_occ = tf.reshape(mo_occ, [-1, meta[2], n_vec_dof])
        # nframe x M x nvec_dof
        prod_occ = tf.matmul(e_occ, mo_occ, transpose_a = True)
        # nframe x (M x nvec_dof)
        prod_occ = tf.reshape(prod_occ, [-1, n_filter * n_vec_dof])
        #
        # nframes x nvir -> (nframes x nvir) x M
        e_vir, weight_l2_vir = self.ds_layer(e_vir, filter_neuron, name = 'vir', reuse = reuse, seed = seed)
        # nframe x nvir x M
        e_vir = tf.reshape(e_vir, [-1, meta[3], n_filter])
        # nframe x nvir x nvec_dof
        mo_vir = tf.reshape(mo_vir, [-1, meta[3], n_vec_dof])
        # nframe x M x nvec_dof
        prod_vir = tf.matmul(e_vir, mo_vir, transpose_a = True)
        # nframe x (M x nvec_dof)
        prod_vir = tf.reshape(prod_vir, [-1, n_filter * n_vec_dof])
        #
        # nframe x (2 x M x (natm x nproj))
        mo_atom = tf.concat([prod_occ, prod_vir], 1)
        # nframe x 2 x M x natm x nproj
        mo_atom = tf.reshape(mo_atom, [-1, 2, n_filter, meta[0], meta[4]])
        # nframe x natm x 2 x M x nproj
        mo_atom = tf.transpose(mo_atom, [0, 3, 1, 2, 4])
        # (nframe x natm) x (2 x M x nproj)
        mo_atom = tf.reshape(mo_atom, [-1, 2 * n_filter * self.nproj])

        weight_l2 = 0
        layer,l2 = self._one_layer(mo_atom, 
                                   self.n_neuron[0], 
                                   name='layer_0', 
                                   reuse = reuse, 
                                   seed = seed)
        weight_l2 += l2
        for ii in range(1,len(self.n_neuron)) :
            if self.resnet and self.n_neuron[ii] == self.n_neuron[ii-1]:
                tl,l2 = self._one_layer(layer, 
                                         self.n_neuron[ii], 
                                         name='layer_'+str(ii), 
                                         reuse = reuse, 
                                         with_dt = True, 
                                         seed = seed)
            else :
                tl,l2 = self._one_layer(layer, 
                                         self.n_neuron[ii], 
                                         name='layer_'+str(ii), 
                                         reuse = reuse, 
                                         with_dt = False, 
                                         seed = seed) 
            layer += tl
            weight_l2 += l2
        # build final layer
        yy_,l2 = self._final_layer(layer, 
                                1, 
                                activation_fn = None, 
                                name='layer_sys_ener', 
                                reuse = reuse, 
                                seed = seed)
        weight_l2 += l2
        # (nframe x natm) x 1
        yy_ = tf.reshape(yy_, [-1, meta[0]])
        # test_ener = tf.reshape (yy_,
        #                         [-1],
        #                         name='o_sys_ener')
        test_ener = tf.reduce_sum(yy_, axis = 1, name = 'o_sys_ener')
        return test_ener, weight_l2 + weight_l2_vir + weight_l2_occ
        

    def ds_layer(self, 
                 i_e, 
                 filter_neuron,
                 filter_resnet_dt=False,
                 activation_fn=tf.nn.tanh, 
                 stddev=1.0,
                 bavg=0.0,
                 name='ds_layer', 
                 reuse=None,
                 seed=None):        
        xyz_scatter = tf.reshape(i_e, [-1,1])
        outputs_size = [1] + filter_neuron
        n_filter = outputs_size[-1]
        weight_l2 = 0
        with tf.variable_scope(name, reuse=reuse):
            for ii in range(1, len(outputs_size)):
                w = tf.get_variable('matrix_'+str(ii), 
                                    [outputs_size[ii-1], outputs_size[ii]], 
                                    tf.float64,
                                    tf.random_normal_initializer(stddev=stddev/np.sqrt(outputs_size[ii]+outputs_size[ii-1]), seed = seed))
                b = tf.get_variable('bias_'+str(ii), 
                                    [1, outputs_size[ii]], 
                                    tf.float64,
                                    tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed))
                weight_l2 += tf.reduce_mean(tf.square(w)) + tf.reduce_mean(tf.square(b))
                if filter_resnet_dt :
                    idt = tf.get_variable('idt_'+str(ii), 
                                          [1, outputs_size[ii]], 
                                          tf.float64,
                                          tf.random_normal_initializer(stddev=0.001, mean = 1.0, seed = seed))
                    weight_l2 += tf.reduce_mean(tf.square(idt))
                if outputs_size[ii] == outputs_size[ii-1]:
                    if filter_resnet_dt :
                        xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b) * idt
                    else :
                        xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b)
                elif outputs_size[ii] == outputs_size[ii-1] * 2: 
                    if filter_resnet_dt :
                        xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b) * idt
                    else :
                        xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b)
                else :
                    xyz_scatter = activation_fn(tf.matmul(xyz_scatter, w) + b)
            xyz_scatter = tf.reshape(xyz_scatter, (-1, n_filter))
        return xyz_scatter, weight_l2


    def _one_layer(self, 
                   inputs, 
                   outputs_size, 
                   activation_fn=tf.nn.tanh, 
                   stddev=1.0,
                   bavg=0.0,
                   name='linear', 
                   reuse=None,
                   seed=None,
                   with_dt = False):
        weight_l2 = 0
        with tf.variable_scope(name, reuse=reuse):
            shape = inputs.get_shape().as_list()
            initer_w = tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+outputs_size), seed = seed)
            initer_b = tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed)
            w = tf.get_variable('matrix', 
                                [shape[1], outputs_size], 
                                tf.float64,
                                initer_w)
            b = tf.get_variable('bias', 
                                [outputs_size], 
                                tf.float64,
                                initer_b)
            weight_l2 += tf.reduce_mean(tf.square(w)) + tf.reduce_mean(tf.square(b))
            hidden = tf.matmul(inputs, w) + b
            if activation_fn != None and with_dt :
                initer_t = tf.random_normal_initializer(stddev=0.001, mean = 0.1, seed = seed)
                timestep=tf.get_variable('timestep',
                                         [outputs_size],
                                         tf.float64,
                                         initer_t)
                weight_l2 += tf.reduce_mean(tf.square(timestep))
        if activation_fn != None:
            if with_dt :
                return activation_fn(hidden) * timestep, weight_l2
            else :
                return activation_fn(hidden), weight_l2
        else:
            return hidden, weight_l2


    def _final_layer(self, 
                     inputs, 
                     outputs_size, 
                     activation_fn=tf.nn.tanh, 
                     stddev=1.0,
                     bavg=-0.0,
                     name='linear', 
                     reuse=None,
                     seed=None):
        with tf.variable_scope(name, reuse=reuse):
            shape = inputs.get_shape().as_list()
            initer_w = tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+outputs_size), seed = seed)
            w = tf.get_variable('matrix', 
                                [shape[1], outputs_size], 
                                tf.float64,
                                initer_w)
            hidden = tf.matmul(inputs, w)
            weight_l2 = tf.reduce_mean(tf.square(w))
        if activation_fn != None:
            return activation_fn(hidden)
        else:
            return hidden, weight_l2
    
