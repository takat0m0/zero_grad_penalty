# -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from model import Model
from data_sample import gaussian_mixture, gaussian_mixture_B, single_gaussian
from util import plot_scatter, plot_scatter2, plot_scatter3
from histogram import plot_histogram

def draw_from_pz(batch_size, z_dim):
    batch_z = np.random.rand(batch_size * z_dim).astype(np.float32)
    ret = np.reshape(batch_z, [batch_size, z_dim]).astype(np.float32)
    return ret


if __name__ == u'__main__':
    
    tf.set_random_seed(0)
    np.random.seed(0)

    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', default = None, type = str)
    parser.add_argument('--mode', '-m', default = None, type = str)    
    args = parser.parse_args()
    
    # parameter
    epoch_num = 230
    z_dim = 4
    batch_size = 256
    num_one_epoch = 1

    target_mu = 3.0
    target_sigma = 0.1
    
    # make model
    print('-- make model --')

    model = Model(z_dim, args.mode)
    model.set_model()

    # training
    print('-- begin training --')
    fixed = np.asarray([[_ * 0.01] for _ in range(501)], dtype = np.float32)
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
            
        for epoch in range(epoch_num):
            print('** epoch {} begin **'.format(epoch))
            #target_sigma *= 0.99
            g_obj = 0.0
            d_obj = 0.0
            
            # plot p_g
            batch_z = draw_from_pz(10000, z_dim)
            tmp = model.generate(sess, batch_z)
            tmp = np.reshape(tmp, [10000])
            #tmp2 = np.random.normal(target_mu, target_sigma, [10000]).astype(np.float32)
            tmp2 = np.asarray([target_mu] * 10000).astype(np.float32)
            plot_histogram(tmp, tmp2, '{}/{}.png'.format(args.dir, epoch))
            val, grad = model.get_disc_value(sess, fixed)

            val = np.reshape(val, [-1])
            grad = np.reshape(grad, [-1])
            print(np.mean(tmp), grad[300])
            plot_scatter2(fixed, grad, '{}_grad'.format(args.dir), '{}'.format(epoch), color = 'red')
            plot_scatter3(fixed, np.log(val), '{}_val'.format(args.dir), '{}'.format(epoch), color = 'blue')
            for step in range(num_one_epoch):
                
                # draw from p_z
                batch_z = draw_from_pz(batch_size, z_dim)

                # draw from p_data
                #batch_inputs = gaussian_mixture(batch_size)
                #batch_inputs = np.random.normal(target_mu, target_sigma, [batch_size, 1]).astype(np.float32)
                batch_inputs = np.asarray([[target_mu]]*batch_size, dtype = np.float32)
                
                #batch_inputs = gaussian_mixture_B(batch_size)
                #batch_inputs = single_gaussian(batch_size)
                
                # train discriminator
                d_obj += model.training_disc(sess, batch_z, batch_inputs)

                # train generator
                g_obj += model.training_gen(sess,  batch_z, batch_inputs)
            '''    
            print('epoch:{}, d_obj = {}, g_obj = {}'.format(epoch,
                                                            d_obj/num_one_epoch,
                                                            g_obj/num_one_epoch))
            
            saver.save(sess, './model.dump')
            '''
