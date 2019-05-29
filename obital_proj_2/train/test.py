#!/usr/bin/env python3

import re
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

kbT = (8.617343E-5) * 300 
beta = 1.0 / kbT
f_cvt = 96.485
cv_dim = 2

def load_graph(frozen_graph_filename, 
               prefix = 'load'):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name=prefix, 
            producer_op_list=None
        )
    return graph

def test_e (sess, sys_meta, t_mo_occ, t_mo_vir, t_e_occ, t_e_vir) :
    graph = sess.graph

    input_meta = graph.get_tensor_by_name ('load/input_meta:0')
    input_e_occ = graph.get_tensor_by_name ('load/input_e_occ:0')
    input_e_vir = graph.get_tensor_by_name ('load/input_e_vir:0')
    input_mo_occ = graph.get_tensor_by_name ('load/input_mo_occ:0')
    input_mo_vir = graph.get_tensor_by_name ('load/input_mo_vir:0')
    o_sys_ener= graph.get_tensor_by_name ('load/o_sys_ener:0')

    feed_dict_test = {input_meta: sys_meta,
                      input_mo_occ: t_mo_occ,
                      input_mo_vir: t_mo_vir,
                      input_e_occ: t_e_occ,
                      input_e_vir: t_e_vir}

    data_ret = sess.run ([o_sys_ener], 
                         feed_dict = feed_dict_test)
    return data_ret[0]


def compute_std (forces) :
    nmodels = forces.shape[0]
    nframes = forces.shape[1]
    ncomps = forces.shape[2]
    
    stds = []
    for ii in range (nframes) :
        # print ( forces[0, ii], forces[1, ii], forces[2, ii], forces[3, ii])
        avg_std = 0
        for jj in range (ncomps) :
            mystd = np.std (forces[:, ii, jj])
            avg_std += mystd * mystd
        avg_std = np.sqrt (avg_std / float(ncomps))
        stds.append (avg_std)
    return np.array (stds)
    

def _main () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", default='frozen_model.pb', type=str,
                        help="Frozen models file to test")
    parser.add_argument("-d", "--data", default='data', type=str,
                        help="The data for test")
    parser.add_argument("-e", "--with-ener", action = 'store_true',
                        help="The use ener")
    args = parser.parse_args()

    models = args.models    
    data_path = args.data

    sys_meta = np.loadtxt(os.path.join(args.data,'system.raw'), dtype = int).reshape([-1])
    natm = sys_meta[0]
    nao = sys_meta[1]
    nocc = sys_meta[2]
    nvir = sys_meta[3]
    nproj = sys_meta[4]
    tr_data_emp2 = np.loadtxt(os.path.join(data_path,'e_mp2.raw')).reshape([-1])
    nframes = tr_data_emp2.shape[0]
    tr_data_dist = np.loadtxt(os.path.join(data_path,'dist.raw')).reshape([-1])
    tr_data_dist = np.ones(tr_data_dist.shape)
    assert(nframes == tr_data_dist.shape[0])
    tr_data_mo_occ = np.loadtxt(os.path.join(data_path,'coeff_occ.raw')).reshape([nframes,nocc*natm*nproj]).reshape([-1])
    tr_data_mo_vir = np.loadtxt(os.path.join(data_path,'coeff_vir.raw')).reshape([nframes,nvir*natm*nproj]).reshape([-1])
    tr_data_e_occ = np.loadtxt(os.path.join(data_path,'ener_occ.raw')).reshape([nframes,nocc]).reshape([-1])
    tr_data_e_vir = np.loadtxt(os.path.join(data_path,'ener_vir.raw')).reshape([nframes,nvir]).reshape([-1])

    # tr_data_emp2 = np.loadtxt(os.path.join(data_path,'e_mp2.raw')).reshape([-1])
    # nframes = tr_data_emp2.shape[0]
    # tr_data_dist = np.loadtxt(os.path.join(data_path,'dist.raw')).reshape([-1])
    # tr_data_dist = np.ones(tr_data_dist.shape)
    # assert(nframes == tr_data_dist.shape[0])
    # tmp_coeff = np.loadtxt(os.path.join(data_path,'mo_coeff.raw')).reshape([nframes,2,2,-1])
    # tmp_ener  = np.loadtxt(os.path.join(data_path,'mo_ener.raw')) .reshape([nframes,2,2,-1])
    # tr_data_mo_occ = tmp_coeff[:,:,0,:].reshape([nframes,-1])
    # tr_data_mo_vir = tmp_coeff[:,:,1,:].reshape([nframes,-1])
    # tr_data_e_occ = tmp_ener[:,:,0,:].reshape([nframes,-1])
    # tr_data_e_vir = tmp_ener[:,:,1,:].reshape([nframes,-1])

    graph = load_graph (models)
    with tf.Session(graph = graph) as sess:        
        ee = test_e (sess,                     
                     sys_meta, 
                     tr_data_mo_occ, 
                     tr_data_mo_vir,
                     tr_data_e_occ, 
                     tr_data_e_vir)
        
    ee = np.reshape(ee, [-1,1])
    tr_data_emp2 = tr_data_emp2.reshape([-1,1])
    print('# ener std: ' + str(np.std(tr_data_emp2)))
    ret = np.concatenate((tr_data_emp2, ee), axis = 1)
    np.savetxt('test.out', ret)

    diff = ee - tr_data_emp2
    diff2 = diff*diff
    print(np.sqrt(np.average(diff * diff)))


if __name__ == '__main__':
    _main()
