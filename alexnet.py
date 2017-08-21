import os
import sys
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
import pdb
from datetime import datetime
from math import ceil
import random
import cv2

from tensorflow.python.framework import ops


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        self.num_calls += 1
        return y
    
flip_gradient = FlipGradientBuilder()


class Alexnet(object):
    def __init__(self, net_path, hash_bit_num):
        '''
        Args:
            net_data: pretrain model (float32)
        '''
        self._net_data = np.load(net_path).item()
        self._initialized = False
        self._lr_mult = None
        self._out_dim = hash_bit_num

    def extract(self, img, train_phase=True):
        '''
        alexnet structure
        Args:
            img: [batch_size, w, h, c] 4-D tensor
        Return:
            hash bits: [batch_size, self._output_dim] tensor
        '''
        lr_mult = dict()
        net_data = self._net_data
        self._initialized = True
        ### Conv1
        ### Output 96, kernel 11, stride 4
        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights', initializer=net_data['conv1'][0])
            conv = tf.nn.conv2d(img, kernel, [1, 4, 4, 1], padding='VALID')
            biases = tf.get_variable('biases', initializer=net_data['conv1'][1])
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        
        ### Pool1
        self.pool1 = tf.nn.max_pool(self.conv1,
                                    ksize=[1,3,3,1],
                                    strides=[1,2,2,1],
                                    padding='VALID',
                                    name='pool1')
        ### LRN1
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn1 = tf.nn.local_response_normalization(self.pool1,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)

        ### Conv2
        ### Output 256, pad 2, kernel 5, group 2
        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('weights', initializer=net_data['conv2'][0])
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(self.lrn1, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(output_groups, 3)
            biases = tf.get_variable('biases', initializer=net_data['conv2'][1])
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
            
        ### Pool2
        self.pool2 = tf.nn.max_pool(self.conv2,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool2')
        ### LRN2
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn2 = tf.nn.local_response_normalization(self.pool2,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)

        ### Conv3
        ### Output 384, pad 1, kernel 3
        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable('weights', initializer=net_data['conv3'][0])
            conv = tf.nn.conv2d(self.lrn2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=net_data['conv3'][1])
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        ### Conv4
        ### Output 384, pad 1, kernel 3, group 2
        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights', initializer=net_data['conv4'][0])
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(self.conv3, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(output_groups, 3)
            biases = tf.get_variable('biases', initializer=net_data['conv4'][1])
            out = tf.nn.bias_add(conv, biases)
            self.conv4 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        ### Conv5
        ### Output 256, pad 1, kernel 3, group 2
        with tf.variable_scope('conv5') as scope:
            kernel = tf.get_variable('weights', initializer=net_data['conv5'][0])
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(self.conv4, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(output_groups, 3)
            biases = tf.get_variable('biases', initializer=net_data['conv5'][1])
            out = tf.nn.bias_add(conv, biases)
            self.conv5 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        ### Pool5
        self.pool5 = tf.nn.max_pool(self.conv5,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool5')
        ### FC6
        ### Output 4096
        with tf.variable_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc6w = tf.get_variable('weights', initializer=net_data['fc6'][0])
            fc6b = tf.get_variable('biases', initializer=net_data['fc6'][1])
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            self.fc5 = pool5_flat
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            if train_phase:
                self.fc6 = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)
            else:
                self.fc6 = tf.nn.relu(fc6l)
            lr_mult[fc6w] = 1
            lr_mult[fc6b] = 2
            
        ### FC7
        ### Output 4096
        with tf.variable_scope('fc7') as scope:
            fc7w = tf.get_variable('weights', initializer=net_data['fc7'][0])
            fc7b = tf.get_variable('biases', initializer=net_data['fc7'][1])
            fc7l = tf.nn.bias_add(tf.matmul(self.fc6, fc7w), fc7b)
            if train_phase:
                self.fc7 = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)
            else:
                self.fc7 = tf.nn.relu(fc7l)
            lr_mult[fc7w] = 1
            lr_mult[fc7b] = 2
        ### Hash Layer
        ### Output hash_bit_num
        with tf.variable_scope("fc8") as scope:
            fc8w = tf.get_variable("weights", [self.fc7.get_shape()[1], self._out_dim],
                initializer=tf.random_normal_initializer(0, 0.01))
            fc8b = tf.get_variable("biases",
                initializer=tf.zeros([self._out_dim]))
            self.fc8 = tf.nn.bias_add(tf.matmul(self.fc7, fc8w), fc8b)
            lr_mult[fc8w] = 10
            lr_mult[fc8b] = 20
        self._lr_mult = lr_mult
        return self.fc8

    @property
    def lr_mult(self):
        assert self._initialized == True, "Alexnet not initialized"
        return self._lr_mult

    @property
    def var_list(self):
        return self._lr_mult.keys()

    @property
    def output_dim(self):
        return self._out_dim

class AdversarialNet(object):
    def __init__(self, input_dim):
        '''
        Args:
            net_data: pretrain model (float32)
        '''
        self._initialized = False
        self._lr_mult = None
        self._input_dim = input_dim

    def extract(self, fc7, l_ad, train_phase=True):
        '''
        alexnet structure
        Args:
            img: [batch_size, w, h, c] 4-D tensor
        Return:
            hash bits: [batch_size, self._output_dim] tensor
        '''
        lr_mult = dict()
        self._initialized = True
        self.feat_fc7 = flip_gradient(fc7, l_ad)
        ### Ad1
        with tf.variable_scope('ad_layer1') as scope:
            weights_ad1 = tf.get_variable('weights', [self._input_dim, 1024], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            biases_ad1 = tf.get_variable('biases', initializer=tf.zeros([1024]))
            ad1 = tf.nn.bias_add(tf.matmul(self.feat_fc7, weights_ad1), biases_ad1)
            if train_phase:
                self.ad1 = tf.nn.dropout(tf.nn.relu(ad1), 0.5)
            else:
                self.ad1 = tf.nn.relu(ad1)
            lr_mult[weights_ad1] = 10
            lr_mult[biases_ad1] = 20

        with tf.variable_scope('ad_layer2') as scope:
            weights_ad2 = tf.get_variable('weights', [self.ad1.get_shape()[1], 1024], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            biases_ad2 = tf.get_variable('biases', initializer=tf.zeros([1024]))
            ad2 = tf.nn.bias_add(tf.matmul(self.ad1, weights_ad2), biases_ad2)
            if train_phase:
                self.ad2 = tf.nn.dropout(tf.nn.relu(ad2), 0.5)
            else:
                self.ad2 = tf.nn.relu(ad2)
            lr_mult[weights_ad2] = 10
            lr_mult[biases_ad2] = 20


        with tf.variable_scope('ad_layer3') as scope:
            weights_ad3 = tf.get_variable('weights', [self.ad2.get_shape()[1], 2], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.3))
            biases_ad3 = tf.get_variable('biases', initializer=tf.zeros([2]))
            self.ad3 = tf.nn.bias_add(tf.matmul(self.ad2, weights_ad3), biases_ad3)
            lr_mult[weights_ad3] = 10
            lr_mult[biases_ad3] = 20

        self._lr_mult = lr_mult
        return self.ad3

    @property
    def lr_mult(self):
        assert self._initialized == True, "Alexnet not initialized"
        return self._lr_mult

    @property
    def var_list(self):
        return self._lr_mult.keys()

    @property
    def output_dim(self):
        return self._out_dim
