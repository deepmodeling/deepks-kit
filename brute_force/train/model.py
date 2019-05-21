#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,time,sys
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

natoms = 2

class Reader(object):
    def __init__(self, config, seed = None):
        # copy from config
        self.data_path = config.data_path
        self.num_epoch = config.num_epoch
        self.batch_size = config.batch_size   
        np.random.seed(seed)

    def prepare(self):
        self.index_count_all = 0
        self.tr_data_emp2 = np.loadtxt(os.path.join(self.data_path,'e_mp2.raw')).reshape([-1])
        nframes = self.tr_data_emp2.shape[0]
        self.tr_data_dist = np.loadtxt(os.path.join(self.data_path,'dist.raw')).reshape([-1])
        self.tr_data_dist = np.ones(self.tr_data_dist.shape)
        assert(nframes == self.tr_data_dist.shape[0])
        tmp_coeff = np.loadtxt(os.path.join(self.data_path,'mo_coeff.raw')).reshape([nframes,2,2,-1])
        tmp_ener  = np.loadtxt(os.path.join(self.data_path,'mo_ener.raw')) .reshape([nframes,2,2,-1])
        self.tr_data_mo_occ = tmp_coeff[:,:,0,:].reshape([nframes,-1])
        self.tr_data_mo_vir = tmp_coeff[:,:,1,:].reshape([nframes,-1])
        self.tr_data_e_occ = tmp_ener[:,:,0,:].reshape([nframes,-1])
        self.tr_data_e_vir = tmp_ener[:,:,1,:].reshape([nframes,-1])
        self.tr_data_e_occ -= np.tile(np.reshape(self.tr_data_e_occ[:,0], [-1,1]), (1, self.tr_data_e_occ.shape[1]))
        self.tr_data_e_vir -= np.tile(np.reshape(self.tr_data_e_occ[:,0], [-1,1]), (1, self.tr_data_e_occ.shape[1]))
        self.train_size_all = self.tr_data_emp2.shape[0]
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
            self.tr_data_mo_occ[ind, :], \
            self.tr_data_mo_vir[ind, :], \
            self.tr_data_e_occ[ind, :], \
            self.tr_data_e_vir[ind, :]

    def sample_train(self, cat = True) :
        return self._sample_train_all()

    def sample_all(self) :
        return \
            self.tr_data_emp2, \
            1./self.tr_data_dist, \
            self.tr_data_mo_occ[:, :], \
            self.tr_data_mo_vir[:, :], \
            self.tr_data_e_occ[:, :], \
            self.tr_data_e_vir[:, :]

    def get_train_size(self) :
        return self.train_size_all

    def get_batch_size(self) :
        return self.batch_size

    def get_data(self):
        return 1./self.tr_data_dist, self.tr_data_mo_occ, self.tr_data_mo_vir, self.tr_data_e_occ, self.tr_data_e_vir

    def mo_dim(self):
        return self.tr_data_mo_occ.shape[1]

    def e_dim(self):
        return self.tr_data_e_occ.shape[1]


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


    def test_error (self, t_emp2, t_dist, t_mo_occ, t_mo_vir, t_e_occ, t_e_vir) :
        ret = self.sess.run([self.l2_loss, self.emp2], 
                            feed_dict={self.emp2_ref: t_emp2,
                                       self.dist: t_dist,
                                       self.mo_occ: t_mo_occ,
                                       self.mo_vir: t_mo_vir,
                                       self.e_occ: t_e_occ,
                                       self.e_vir: t_e_vir,
                                       self.is_training: False})
        np.savetxt('tmp.out', np.concatenate((t_emp2, ret[1])).reshape(2,-1).T)
        error = np.sqrt(ret[0])
        # print(ret[1], t_emp2)
        return error


    def train(self, reader, seed = None):

        reader.prepare()
        self.mo_dim = reader.mo_dim()
        self.e_dim = reader.e_dim()
        self.natoms = natoms
        self.ntests = natoms
        # self.nao2 = self.mo_dim // (self.natoms * self.ntests)
        self.mo_dim_test = self.mo_dim // self.ntests
        
        # placeholders
        self.dist = tf.placeholder(tf.float64, [None], name='input_dist')
        self.mo_occ = tf.placeholder(tf.float64, [None, self.mo_dim], name='input_mo_occ')
        self.mo_vir = tf.placeholder(tf.float64, [None, self.mo_dim], name='input_mo_vir')
        self.e_occ = tf.placeholder(tf.float64, [None, self.e_dim], name='input_e_occ')
        self.e_vir = tf.placeholder(tf.float64, [None, self.e_dim], name='input_e_vir')
        self.emp2_ref = tf.placeholder(tf.float64, [None], name='input_emp2')
        self.is_training = tf.placeholder(tf.bool)

        # compute statistic 
        data_stat = self.compute_statistic(reader)

        # build dnn 
        self.emp2 \
            = self.build_dnn(self.dist, 
                             self.mo_occ, 
                             self.mo_vir, 
                             self.e_occ, 
                             self.e_vir, 
                             suffix = 'test',
                             stat = data_stat, 
                             reuse = False, 
                             seed = seed)
        self.l2_loss \
            = self.build_loss(self.emp2_ref, self.emp2, suffix = 'test') 

        # learning rate
        self._extra_train_ops = []
        self.global_step \
            = tf.get_variable('global_step', 
                              [],
                              initializer = tf.constant_initializer(1),
                              trainable = False, 
                              dtype = tf.int32)
        self.global_epoch = self.global_step * reader.get_batch_size() // reader.get_train_size()
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
        t_emp2, t_dist, t_mo_occ, t_mo_vir, t_e_occ, t_e_vir = reader.sample_train()
        ta_emp2, ta_dist, ta_mo_occ, ta_mo_vir, ta_e_occ, ta_e_vir = reader.sample_all()
        error = self.test_error(t_emp2, t_dist, t_mo_vir, t_mo_vir, t_e_vir, t_e_vir)
        current_lr = self.sess.run(tf.to_double(self.learning_rate))
        if self.display_in_training:
            print ("epoch: %3u, ab_err: %.4e, lr: %.4e" % (epoch_used, error, current_lr))

        # training
        while epoch_used < reader.num_epoch:
            d_emp2, d_dist, d_mo_occ, d_mo_vir, d_e_occ, d_e_vir = reader.sample_train()
            # exec training op
            self.sess.run([self.train_op], 
                          feed_dict={self.emp2_ref: d_emp2,
                                     self.dist: d_dist,
                                     self.mo_occ: d_mo_occ,
                                     self.mo_vir: d_mo_vir,
                                     self.e_occ: d_e_occ,
                                     self.e_vir: d_e_vir,
                                     self.is_training: True})
            sample_used += reader.get_batch_size()
            if (sample_used // reader.get_train_size()) > epoch_used:
                epoch_used = sample_used // reader.get_train_size()
                if epoch_used % self.n_displayepoch == 0:
                    save_path = saver.save(self.sess, os.getcwd() + "/" + "model.ckpt")
                    error = self.test_error(d_emp2, d_dist, d_mo_occ, d_mo_vir, d_e_occ, d_e_vir)
                    error_a = self.test_error(ta_emp2, ta_dist, ta_mo_occ, ta_mo_vir, ta_e_occ, ta_e_vir)
                    current_lr = self.sess.run(tf.to_double(self.learning_rate))
                    if self.display_in_training:
                        print ("epoch: %3u, ab_err: %.4e, ab_err_all: %.4e, lr: %.4e" % (epoch_used, error, error_a, current_lr))
                        sys.stdout.flush()

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
                   suffix = None) :
        ee_diff = ener_ref - ener
        ee_diff_norm = ee_diff * ee_diff
        l2_loss = tf.reduce_mean(tf.square(ener_ref - ener), 
                                 name='l2_loss_' + suffix)
        return l2_loss

    
    def scale_var(self, 
                  mo_occ, 
                  shift,
                  scale, 
                  name ) :
        assert(shift.size == scale.size)
        t_mo_occ_a\
            = tf.get_variable(name + '_a', 
                              [shift.size],
                              dtype = tf.float64,
                              trainable = False,
                              initializer = tf.constant_initializer(shift))
        t_mo_occ_s\
            = tf.get_variable(name + '_s', 
                              [scale.size],
                              dtype = tf.float64,
                              trainable = False,
                              initializer = tf.constant_initializer(scale))
        mo_occ = (mo_occ - t_mo_occ_a) * t_mo_occ_s
        return mo_occ
    

    def build_dnn (self, 
                   dist,
                   mo_occ,
                   mo_vir,
                   e_occ,
                   e_vir,
                   stat = None,
                   suffix = None,
                   reuse = None, 
                   seed = None):        
        if stat is not None :
            mo_occ = self.scale_var(mo_occ, stat[0], stat[1], 'mo_occ')
            mo_vir = self.scale_var(mo_vir, stat[2], stat[3], 'mo_vir')
            e_occ = self.scale_var(e_occ, stat[4], stat[5], 'e_occ')
            e_vir = self.scale_var(e_vir, stat[6], stat[7], 'e_vir')
        # test_ener = self.build_atom_net(mo_occ, mo_vir, reuse, seed)
        # sys_ener = tf.reduce_sum(test_ener, axis = 1, name = 'o_ener_' + suffix)
        sys_ener = self.build_system_net(dist, mo_occ, mo_vir, e_occ, e_vir, reuse, seed)
        sys_ener = tf.identity(sys_ener, name = 'sys_ener_' + suffix)
        # l2 loss
        return sys_ener

    def build_system_net(self, 
                         dist,
                         mo_occ, 
                         mo_vir, 
                         e_occ, 
                         e_vir, 
                         reuse = None, 
                         seed = None, 
                         use_ds_layer = True):        
        if use_ds_layer :
            filter_neuron = [5,10,10]
            # mo_occ = self.ds_layer(mo_occ, filter_neuron, name = 'occ', reuse = reuse, seed = seed)
            # mo_occ = self.ds_layer(mo_occ, filter_neuron, name = 'mo_occ', reuse = reuse, seed = seed)
            # e_occ  = self.ds_layer(e_occ,  filter_neuron, name = 'e_occ',  reuse = reuse, seed = seed)
            # mo_vir = self.ds_layer(mo_vir, filter_neuron, name = 'mo_vir', reuse = reuse, seed = seed)
            # e_vir  = self.ds_layer(e_vir,  filter_neuron, name = 'e_vir',  reuse = reuse, seed = seed)        
        dist = tf.reshape(dist, [-1,1])
        mo_atom = tf.concat([dist, mo_occ, mo_vir, e_occ, e_vir], 1)
        # mo_atom = tf.concat([e_occ, e_vir], 1)
        # mo_atom = tf.concat([mo_occ, mo_vir], 1)
        # normalize input
        # build hidden layers
        layer = self._one_layer(mo_atom, 
                                self.n_neuron[0], 
                                name='layer_0', 
                                reuse = reuse, 
                                seed = seed)
        for ii in range(1,len(self.n_neuron)) :
            if self.resnet and self.n_neuron[ii] == self.n_neuron[ii-1]:
                layer += self._one_layer(layer, 
                                         self.n_neuron[ii], 
                                         name='layer_'+str(ii), 
                                         reuse = reuse, 
                                         with_dt = True, 
                                         seed = seed)
            else :
                layer  = self._one_layer(layer, 
                                         self.n_neuron[ii], 
                                         name='layer_'+str(ii), 
                                         reuse = reuse, 
                                         with_dt = False, 
                                         seed = seed) 
        # build final layer
        yy_ = self._final_layer(layer, 
                                1, 
                                activation_fn = None, 
                                name='layer_sys_ener', 
                                reuse = reuse, 
                                seed = seed)
        test_ener = tf.reshape (yy_,
                                [-1],
                                name='o_sys_ener')
        return test_ener


    def build_atom_net(self, 
                       mo_occ, 
                       mo_vir, 
                       reuse = None, 
                       seed = None, 
                       use_ds_layer = False):        
        mo_occ_atom = tf.reshape(mo_occ, [-1, self.mo_dim_test])
        mo_vir_atom = tf.reshape(mo_vir, [-1, self.mo_dim_test])
        if use_ds_layer :
            filter_neuron = [5,10,20]
            mo_occ_atom = self.ds_layer(mo_occ_atom, filter_neuron, name = 'occ', reuse = reuse, seed = seed)
            mo_vir_atom = self.ds_layer(mo_vir_atom, filter_neuron, name = 'vir', reuse = reuse, seed = seed)
        mo_atom = tf.concat([mo_occ_atom, mo_vir_atom], 1)
        # normalize input
        # build hidden layers
        layer = self._one_layer(mo_atom, 
                                self.n_neuron[0], 
                                name='layer_0', 
                                reuse = reuse, 
                                seed = seed)
        for ii in range(1,len(self.n_neuron)) :
            if self.resnet and self.n_neuron[ii] == self.n_neuron[ii-1]:
                layer += self._one_layer(layer, 
                                         self.n_neuron[ii], 
                                         name='layer_'+str(ii), 
                                         reuse = reuse, 
                                         with_dt = True, 
                                         seed = seed)
            else :
                layer  = self._one_layer(layer, 
                                         self.n_neuron[ii], 
                                         name='layer_'+str(ii), 
                                         reuse = reuse, 
                                         with_dt = False, 
                                         seed = seed) 
        # build final layer
        yy_ = self._final_layer(layer, 
                                1, 
                                activation_fn = None, 
                                name='layer_atom_ener', 
                                reuse = reuse, 
                                seed = seed)
        test_ener = tf.reshape (yy_,
                                [-1, self.ntests],
                                name='o_atom_ener')
        return test_ener
        

    def ds_layer(self, 
                 xx, 
                 filter_neuron,
                 filter_resnet_dt=True,
                 activation_fn=tf.nn.tanh, 
                 stddev=1.0,
                 bavg=0.0,
                 name='ds_layer', 
                 reuse=None,
                 seed=None):
        xx_shape = xx.get_shape().as_list()
        xyz_scatter = tf.reshape(xx, [-1,1])
        outputs_size = [1] + filter_neuron
        n_filter = outputs_size[-1]
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
                if filter_resnet_dt :
                    idt = tf.get_variable('idt_'+str(ii), 
                                          [1, outputs_size[ii]], 
                                          tf.float64,
                                          tf.random_normal_initializer(stddev=0.001, mean = 1.0, seed = seed))
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
            xyz_scatter = tf.reshape(xyz_scatter, (-1, xx_shape[1], n_filter))
            xyz_scatter = tf.reduce_sum(xyz_scatter, axis = 1)
            return xyz_scatter


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
            hidden = tf.matmul(inputs, w) + b
            if activation_fn != None and with_dt :
                initer_t = tf.random_normal_initializer(stddev=0.001, mean = 0.1, seed = seed)
                timestep=tf.get_variable('timestep',
                                         [outputs_size],
                                         tf.float64,
                                         initer_t)
        if activation_fn != None:
            if with_dt :
                return activation_fn(hidden) * timestep
            else :
                return activation_fn(hidden)
        else:
            return hidden


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
        if activation_fn != None:
            return activation_fn(hidden)
        else:
            return hidden
    
