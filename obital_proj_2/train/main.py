#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
import tensorflow as tf
import numpy as np
from model import GroupReader, Model

class Config(object):
    batch_size = 64
    n_displayepoch = 100
    # use Batch Normalization or not
    useBN = False#True
    num_epoch = 3000
    n_neuron = [240, 120, 60, 30]
    starter_learning_rate = 0.003
    decay_steps = 10
    decay_rate = 0.96
    data_path = '../../data.sub10/'
    resnet = False
    graph_file = None
    with_ener = True
    
    display_in_training = True

def reset_batch_size (config) :
    for ii in config.data_path :
        tr_data = np.loadtxt(os.path.join(ii,'e_mp2.raw'))
        if tr_data.shape[0] < config.batch_size :
            config.batch_size = tr_data.shape[0]
            print ("using new batch_size of %d" % config.batch_size)

def print_conf (config, nthreads) :
    print ("# num_threads       %d" % nthreads)
    print ("# neurons           " + str(config.n_neuron))
    print ("# batch size        " + str(config.batch_size))
    print ("# num_epoch         " + str(config.num_epoch))
    print ("# lr_0              " + str(config.starter_learning_rate))
    print ("# decay_steps       " + str(config.decay_steps))
    print ("# decay_rate        " + str(config.decay_rate))
    print ("# resnet            " + str(config.resnet))
    print ("# graph_file        " + str(config.graph_file))
    print ("# with ener         " + str(config.with_ener))

def main():
    parser = argparse.ArgumentParser(
        description="*** Train a model. ***")
    parser.add_argument('-t','--numb-threads', type=int, default = 4,
                        help='the number of threads.')
    parser.add_argument('-d','--data-path', type=str, nargs = '+',
                        help='the path to data file data.raw.')
    parser.add_argument('-n','--neurons', type=int, default = [240, 120, 60, 30], nargs='+',
                        help='the number of neurons in each hidden layer.')
    parser.add_argument('-b','--batch-size', type=int, default = 64,
                        help='the batch size.')
    parser.add_argument('-e','--numb-epoches', type=int, default = 3000,
                        help='the number of epoches.')
    parser.add_argument('-l','--starter-lr', type=float, default = 0.003,
                        help='the starter learning rate.')
    parser.add_argument('--decay-steps', type=int, default = 10,
                        help='the decay steps.')
    parser.add_argument('--decay-rate', type=float, default = 0.96,
                        help='the decay rate.')
    parser.add_argument('--with-ener', action = 'store_true',
                        help='if use energy level information.')
    parser.add_argument('--resnet', action = 'store_true',
                        help='try using resNet if two neighboring layers are of the same size.')
    args = parser.parse_args()

    seed = 1

    config = Config()
    config.data_path = args.data_path
    config.n_neuron = args.neurons
    config.batch_size = args.batch_size
    config.num_epoch = args.numb_epoches
    config.starter_learning_rate = args.starter_lr
    config.decay_steps = args.decay_steps
    config.decay_rate = args.decay_rate
    config.resnet = args.resnet
    config.with_ener = args.with_ener
    reset_batch_size (config)
    print_conf (config, args.numb_threads)

    tf.reset_default_graph()
    tf_config = tf.ConfigProto(intra_op_parallelism_threads=args.numb_threads, 
                               inter_op_parallelism_threads=2)

    with tf.Session(config = tf_config) as sess:
        g_reader = GroupReader(config.data_path, config.batch_size, seed = seed)
        model = Model(config, sess)        
        model.train(g_reader, config.num_epoch, seed = seed)

if __name__ == '__main__':
    main()
