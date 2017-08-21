##################################################################################
# 2017 01.16 Created by Shichen Liu                                              #
# Residual Transfer Network implemented by tensorflow                            #
#                                                                                #
#                                                                                #
##################################################################################

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import scipy.io as sio
import time
import pdb
from datetime import datetime
from math import ceil
import random
import cv2
from alexnet import Alexnet
from utils import *
import shutil

IMAGE_SIZE = 227

class Net(object):
    def __init__(self, config):
        ### Initialize setting
        log('setup', 'Initializing network')
        np.set_printoptions(precision=4)
        self.device = config['device']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.n_class = config['n_class']
        self.hash_bit_num = config['hash_bit_num']
        self.max_iter = config['max_iter']
        self.save_dir = config['save_dir']
        self.save_prefix = config['save_prefix']
        self.snapshot = config['snapshot']
        self.quantization_loss_weight = config['quantization_loss_weight']
        self.learning_rate_decay_factor = config['learning_rate_decay_factor']
        self.decay_step = config['decay_step']

        self.mean_file = './data/imagenet_mean.npy'

        ### Setup session
        log('setup', 'Launching session')
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        configProto.log_device_placement=True
        self.sess = tf.Session(config=configProto)
        self.model_weights = config['model_weights']

        ### Construct network structure
        log('setup', 'Creating network')
        with tf.device(self.device):
            ### Setup inputs
            self.train_img = tf.placeholder(tf.float32, 
                [self.batch_size, 256, 256, 3])
            self.train_label = tf.placeholder(tf.float32, 
                [self.batch_size, self.n_class])
            self.global_step = tf.Variable(0, trainable=False)
            ### Construct CNN
            self.cnn = Alexnet(self.model_weights, self.hash_bit_num)
            ### Construct train net
            train_img = self.preprocess_img(self.train_img, self.batch_size, True)
            with tf.variable_scope("cnn"):
                feature = self.cnn.extract(train_img)
                hash_bit = tf.tanh(feature)
                self.hash_bit = hash_bit
            self.lr_mult = self.cnn.lr_mult

            log('setup', 'Construct Losses')
            ### pairwise cross entropy loss
            #inner_product = tf.matmul(hash_bit, tf.transpose(hash_bit))
            #normalized_hash_bit = tf.divide(self.hash_bit, tf.reshape(tf.norm(self.hash_bit, axis=1), [self.batch_size, 1]))
            inner_product = tf.divide(float(self.hash_bit_num) / 2, tf.add(tf.reduce_sum(tf.square(tf.subtract(tf.reshape(hash_bit, [self.batch_size, 1, self.hash_bit_num]), tf.reshape(hash_bit, [1, self.batch_size, self.hash_bit_num]))), axis=2), 1.0))
            similarity = tf.clip_by_value(tf.matmul(self.train_label, tf.transpose(self.train_label)), 0.0, 1.0)
            batch_n = tf.shape(hash_bit)[0]
            t_ones = tf.ones([batch_n, batch_n])
            exact_pairwise_loss = tf.subtract(tf.log(tf.add(t_ones, tf.exp(inner_product))), tf.multiply(similarity, inner_product))
            approximate_pairwise_loss = tf.multiply(inner_product, tf.subtract(1.0, similarity))
            threshold_p = tf.less(tf.abs(inner_product), 15.0)
            self.pairwise_loss = tf.reduce_mean(tf.where(threshold_p, exact_pairwise_loss, approximate_pairwise_loss))

            ### quantization loss
            one_bias = tf.subtract(tf.abs(hash_bit), 1.0)
            exact_quantization_loss = tf.log(tf.div(tf.add(tf.exp(one_bias), tf.exp(-one_bias)), 2.0))
            approximate_quantization_loss = tf.subtract(one_bias, tf.log(2.0))
            threshold_q = tf.less(tf.abs(one_bias), 15.0)
            self.quantization_loss = tf.reduce_mean(tf.where(threshold_q, exact_quantization_loss, approximate_quantization_loss))
            
            ### overall loss
            self.loss = tf.add(self.pairwise_loss, tf.multiply(self.quantization_loss, self.quantization_loss_weight))
            self.c_loss = self.pairwise_loss
            self.q_loss = tf.multiply(self.quantization_loss, self.quantization_loss_weight)

            ### compute gradients
            log('setup', 'Set gradients')
            self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step, self.learning_rate_decay_factor, staircase=True)
            opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            grads_and_vars = opt.compute_gradients(self.loss, self.cnn.var_list)

            ### apply gradients
            check = lambda var, w: var if var is not None else tf.zeros(tf.shape(w))
            self.train_op = opt.apply_gradients([
                (check(grads_and_vars[i][0], w) * self.lr_mult[w], w) for (i, w) in enumerate(self.cnn.var_list)],
                global_step=self.global_step)

            ### init all variables
            log('setup', 'Initialize all variables')
            self.sess.run(tf.global_variables_initializer())

    def train(self, train_set):
        log('train', 'training starts')
        for i in xrange(self.max_iter):
            train_img, train_label = train_set.next_batch(self.batch_size)
            _, loss, hash_bit, p_loss, q_loss = self.sess.run([self.train_op, self.loss, self.hash_bit, self.pairwise_loss, self.quantization_loss], feed_dict={self.train_img: train_img, self.train_label: train_label})
            #if (i + 1) % 100 == 0:
                #print hash_bit
            print "iter " + str(i+1) + ", loss " + str(loss) + ", pairwise_loss " + str(p_loss) + ", quantization_loss " + str(q_loss)
            
            ### snapshot model
            if (i + 1) % self.snapshot == 0:
                saver = tf.train.Saver()
                save_path = self.save_dir + self.save_prefix + "_" + str(i + 1)
                shutil.rmtree(save_path, ignore_errors=True)
                os.mkdir(save_path)
                print "snapshot to " + save_path + '/model.ckpt'
                saver.save(self.sess, save_path + '/model.ckpt')
                

    def preprocess_img(self, img, batch_size, train_phase, oversample=False):
        '''
        pre-process input image:
        Args:
            img: 4-D tensor
            batch_size: Int 
            train_phase: Bool
        Return:
            distorted_img: 4-D tensor
        '''
        reshaped_image = tf.cast(img, tf.float32)
        mean = tf.constant(np.load(self.mean_file), dtype=tf.float32, shape=[1, 256, 256, 3])
        reshaped_image -= mean
        crop_height = IMAGE_SIZE
        crop_width = IMAGE_SIZE
        if train_phase:
            distorted_img = tf.stack([tf.random_crop(tf.image.random_flip_left_right(each_image), [crop_height, crop_width, 3]) for each_image in tf.unstack(reshaped_image)])
        else:
            if oversample:
                distorted_img1 = tf.stack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 0, 0, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img2 = tf.stack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 28, 28, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img3 = tf.stack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 28, 0, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img4 = tf.stack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 0, 28, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img5 = tf.stack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 14, 14, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img6 = tf.stack([tf.image.crop_to_bounding_box(each_image, 0, 0, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img7 = tf.stack([tf.image.crop_to_bounding_box(each_image, 28, 28, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img8 = tf.stack([tf.image.crop_to_bounding_box(each_image, 28, 0, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img9 = tf.stack([tf.image.crop_to_bounding_box(each_image, 0, 28, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img0 = tf.stack([tf.image.crop_to_bounding_box(each_image, 14, 14, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])            
                distorted_img = tf.concat(0, [distorted_img1, distorted_img2, distorted_img3, distorted_img4, distorted_img5, distorted_img6, distorted_img7, distorted_img8, distorted_img9, distorted_img0])
            else:
                distorted_img = tf.stack([tf.image.crop_to_bounding_box(each_image, 14, 14, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
        return distorted_img


def train():
    config = dict(
        device = '/gpu:' + sys.argv[1],
        max_iter = 10000,
        batch_size = 64,
        learning_rate = 0.00002,
        decay_step = 500,
        learning_rate_decay_factor = 0.5,
        n_class = 12,
        hash_bit_num = 48,
        quantization_loss_weight = 8,
        save_dir = "./models/",
        save_prefix = "ours33",
        snapshot = 10000,
        model_weights = "./data/pretrain_model.npy"
        )
    net = Net(config)
    train_set = Dataset("./data/challenge/validation_train_list.txt", config['n_class'])
    net.train(train_set)

train()
