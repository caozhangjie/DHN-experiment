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
from models.alexnet import Alexnet
from utils import *

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
        self.test_size = config['test_size']
        self.train_size = config['train_size']
        self.max_iter = config['max_iter']
        self.save_dir = config['save_dir']
        self.test_iter = config['test_iter']
        self.save_prefix = ""

        self.mean_file = './data/mean_file/imagenet_mean.npy'
        self.bottleneck_dim = 256
        self.show_iter = 100

        ### Setup session
        log('setup', 'Launching session')
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        self.sess = tf.Session(config=configProto)
        self.model_weights = config['model_weights']

        ### Construct network structure
        log('setup', 'Creating network')
        with tf.device(self.device):
            ### Setup inputs
            self.source_img = tf.placeholder(tf.float32, 
                [self.batch_size, 256, 256, 3])
            self.target_img = tf.placeholder(tf.float32, 
                [self.batch_size, 256, 256, 3])
            self.test_img = tf.placeholder(tf.float32, 
                [1, 256, 256, 3])
            self.source_label = tf.placeholder(tf.float32, 
                [self.batch_size, self.n_class])
            self.target_label = tf.placeholder(tf.float32, 
                [self.batch_size, self.n_class])
            self.test_label = tf.placeholder(tf.float32, 
                [1, self.n_class])
            self.global_step = tf.Variable(0, trainable=False)
            ### Construct CNN
            self.cnn = Alexnet(self.model_weights)
            ### Construct train net
            source_img = self.preprocess_img(self.source_img, self.batch_size, True)
            target_img = self.preprocess_img(self.target_img, self.batch_size, True)
            with tf.variable_scope("cnn"):
                source_feature = self.cnn.extract(source_img)
                tf.get_variable_scope().reuse_variables()
                target_feature = self.cnn.extract(target_img)
            self.lr_mult = self.cnn.lr_mult
            with tf.variable_scope("classifier"):
                source_fc7, source_fc8 = self.classifier(source_feature)
                tf.get_variable_scope().reuse_variables()
                target_fc7, target_fc8 = self.classifier(target_feature)
            with tf.variable_scope("d_net"):
                dc_source = self.discriminator(self.gradient_lr(source_fc7, high=1.0, max_iter=10000, alpha=10))
                #dc_source = self.discriminator(self.gradient_lr(self.gradient_lr(source_fc7, high=2.0, max_iter=5000, alpha=1), high=2.0, max_iter=5000, alpha=1, reverse=True)+self.gradient_lr(self.gradient_lr(source_fc7, high=0.1, max_iter=5000, alpha=1, minus=False)))
                tf.get_variable_scope().reuse_variables()
                dc_target = self.discriminator(self.gradient_lr(target_fc7, high=1.0, max_iter=10000, alpha=10))
                #dc_target = self.discriminator(self.gradient_lr(self.gradient_lr(target_fc7, high=2.0, max_iter=5000, alpha=1), high=2.0, max_iter=5000, alpha=1, reverse=True)+self.gradient_lr(self.gradient_lr(target_fc7, high=0.1, max_iter=5000, alpha=1, minus=False)))
            log('setup', 'Construct Losses')
            dc_label = tf.concat([tf.ones([self.batch_size, 1]), -tf.ones([self.batch_size, 1])], 0)
            dc_output = tf.concat([dc_source, dc_target], 0)
            source_importance = tf.stop_gradient(tf.exp(-tf.tanh(dc_source)))
            self.srci = source_importance
            self.lp_loss = tf.reduce_mean(tf.multiply(source_importance, tf.nn.softmax_cross_entropy_with_logits(logits=source_fc8, labels=self.source_label)))
            #self.lp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=source_fc8, labels=self.source_label))
            #self.dc_loss = 0.1 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dc_output, labels=dc_label))
            self.dc_loss = 0.1 * tf.reduce_mean(tf.log(1 + tf.exp(tf.multiply(-dc_label, tf.tanh(dc_output)))))
            log('setup', 'Set gradients')
            self.lr_lp = tf.train.exponential_decay(self.learning_rate, -0.75, 1.0, 1.0+0.002*tf.cast(self.global_step, tf.float32))
            opt = tf.train.MomentumOptimizer(learning_rate=self.lr_lp, momentum=0.9)
            grads_and_vars_lp = opt.compute_gradients(self.lp_loss+self.dc_loss, self.cnn.var_list+self.classifier_list+self.d_net_list)
            grads_and_vars_dc = opt.compute_gradients(self.dc_loss, self.d_net_list)
            
            check = lambda var, w: var if var is not None else tf.zeros(tf.shape(w))
            self.train_op_lp = opt.apply_gradients([
                (check(grads_and_vars_lp[i][0], w) * self.lr_mult[w], w) for (i, w) in enumerate(self.cnn.var_list+self.classifier_list+self.d_net_list)],
                global_step=self.global_step)
            self.train_op_dc = opt.apply_gradients([
                (check(grads_and_vars_dc[i][0], w) * self.lr_mult[w], w) for (i, w) in enumerate(self.d_net_list)])
            
            ### Construct test net
            log('setup', 'Construct Test Net')
            test_img = self.preprocess_img(self.test_img, 1, False)
            with tf.variable_scope("cnn", reuse=True):
                test_feature = self.cnn.extract(test_img, train_phase=False)
            with tf.variable_scope("classifier", reuse=True):
                self.test_feature, test_fc8 = self.classifier(test_feature)
            test_output = tf.reduce_mean(test_fc8, 0)
            self.test_output = tf.nn.softmax(test_output)
            self.test_result = tf.equal(tf.argmax(test_output, 0), tf.argmax(self.test_label, 1))
            log('setup', 'Initialize all variables')
            self.sess.run(tf.global_variables_initializer())

    def train(self, source_set, target_set):
        log('train', 'training starts')
        stats = np.zeros([self.n_class])
        for i in xrange(self.max_iter):
            if i % self.test_iter == 0:
                if i % 5000 == 0:
                    self.validate(target_set, self.test_size, './results/source_%s.mat'%i)
                    self.validate(source_set, self.train_size, './results/target_%s.mat'%i)
                else:
                    self.validate(target_set, self.test_size)
                    self.validate(source_set, self.train_size)
            DC_ITER = 0
            LP_ITER = 1
            SH_ITER = 100
#            if i % self.show_iter == 0:
#                DC_ITER = 100
            for j in xrange(DC_ITER):
                source_img, source_label = source_set.next_batch(self.batch_size)
                target_img, target_label = target_set.next_batch(self.batch_size)
                _, dc_loss = self.sess.run([self.train_op_dc, self.dc_loss], feed_dict={
                    self.source_img: source_img,
                    self.target_img: target_img})
            for j in xrange(LP_ITER):
                source_img, source_label = source_set.next_batch(self.batch_size)
                target_img, target_label = target_set.next_batch(self.batch_size)
                _, lp_loss, dc_loss, lr_lp, srci, grl = self.sess.run([self.train_op_lp, self.lp_loss, self.dc_loss, self.lr_lp, self.srci, self.grl], feed_dict={
                    self.source_img: source_img,
                    self.target_img: target_img,
                    self.source_label: source_label})
            stat = lambda x: 0 if len([srci[j][0] for j in xrange(self.batch_size) if source_label[j][x] == 1]) == 0 else np.mean([srci[j][0] for j in xrange(self.batch_size) if source_label[j][x] == 1])
            stats += map(stat, range(self.n_class))
            if i % SH_ITER == 0:
                print stats / SH_ITER
                stats = np.zeros([self.n_class])
                print srci.T
                log('train', '''step = %6d, lr = %.8f
    lp_loss = %.4f, dc_loss = %.4f, grl = %.8f''' % 
                    (i, lr_lp, lp_loss, dc_loss, grl))

    def validate(self, test_set, test_size, save_name=None):
        log('valid', 'validate starts')
        test_img, test_label = test_set.full_data()
        if save_name is not None:
            features = np.zeros([test_size, self.bottleneck_dim])
            outputs = np.zeros([test_size, self.n_class])
            labels = np.zeros([test_size])
        acc = 0.
        for i in xrange(test_size):
            res, feature, output = self.sess.run([self.test_result, self.test_feature, self.test_output], feed_dict={
                self.test_img: np.expand_dims(test_img[i, :], 0),
                self.test_label: np.expand_dims(test_label[i, :], 0)
                })
            acc += res
            if save_name is not None:
                features[i] = feature
                outputs[i] = output
                labels[i] = np.argmax(test_label[i])
        acc /= test_size
        log('valid', 'acc = %.6f' % acc)
        if save_name is not None:
            sio.savemat(save_name, {
                'feature': features,
                'output': outputs,
                'label': labels})
        log('valid', 'validate finished')

    def gradient_lr(self, x, low=0.0, high=1.0, max_iter=2000.0, alpha=10.0, reverse=False):
        height = high - low
        progress = tf.minimum(1.0, tf.cast(self.global_step, tf.float32) / max_iter)
        lr_mult = -(tf.div(2.0*height, (1.0+tf.exp(-alpha*progress))) - height + low)
        #lr_mult = -tf.reduce_max([(1 - 1/10000.*tf.cast(self.global_step, tf.float32))/10000.*tf.cast(self.global_step, tf.float32)*4, 0.])
        self.grl = lr_mult
        def gradient_lr_grad(op, grad):
            x_ = op.inputs[0]
            y_ = op.inputs[1]
            scale = y_ if not reverse else y_ - high
            return [grad * scale, tf.zeros([])]
        with ops.name_scope(None, "Gradient", [x, lr_mult]) as name:
            gr_x_y = py_func(lambda x_, y_: x_,
                [x, lr_mult],
                [tf.float32],
                name=None,
                grad=gradient_lr_grad)
            return gr_x_y[0]
    
    def classifier(self, feature=None, fc7=None):
        if fc7 is None:
            with tf.variable_scope("bottleneck"):
                fc7w = tf.get_variable("w", [self.cnn.output_dim, self.bottleneck_dim],
                    initializer=tf.random_normal_initializer(0, 0.005))
                fc7b = tf.get_variable("b",
                    initializer=tf.scalar_mul(0.1, tf.ones([self.bottleneck_dim])))
                fc7l = tf.nn.bias_add(tf.matmul(feature, fc7w), fc7b)
                fc7 = tf.nn.relu(fc7l)
                self.lr_mult[fc7w] = 10
                self.lr_mult[fc7b] = 20
            with tf.variable_scope("fc8"):
                fc8w = tf.get_variable("w", [self.bottleneck_dim, self.n_class],
                    initializer=tf.random_normal_initializer(0, 0.01))
                fc8b = tf.get_variable("b",
                    initializer=tf.zeros([self.n_class]))
                fc8 = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)
                self.lr_mult[fc8w] = 10
                self.lr_mult[fc8b] = 20
            self.classifier_list = [fc7w, fc7b, fc8w, fc8b]
            return fc7, fc8
        else:
            with tf.variable_scope("fc8"):
                fc8w = tf.get_variable("w", [self.bottleneck_dim, self.n_class],
                    initializer=tf.random_normal_initializer(0, 0.01))
                fc8b = tf.get_variable("b",
                    initializer=tf.zeros([self.n_class]))
                fc8 = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)
                self.lr_mult[fc8w] = 10
                self.lr_mult[fc8b] = 20
            return fc8

    def discriminator(self, fc7):
        DIS_DIM = 1024
        with tf.variable_scope("d_net1"):
            d_net1w = tf.get_variable("w", [self.bottleneck_dim, DIS_DIM],
                initializer=tf.random_normal_initializer(0, 0.01))
            d_net1b = tf.get_variable("b",
                initializer=tf.zeros([DIS_DIM]))
            d_net1l = tf.nn.bias_add(tf.matmul(fc7, d_net1w), d_net1b)
            #d_net1 = tf.nn.dropout(tf.nn.relu(d_net1l), 0.5)
            d_net1 = tf.nn.relu(d_net1l)
            self.lr_mult[d_net1w] = 10
            self.lr_mult[d_net1b] = 20
        with tf.variable_scope("d_net2"):
            d_net2w = tf.get_variable("w", [DIS_DIM, DIS_DIM],
                initializer=tf.random_normal_initializer(0, 0.01))
            d_net2b = tf.get_variable("b",
                initializer=tf.zeros([DIS_DIM]))
            d_net2l = tf.nn.bias_add(tf.matmul(d_net1, d_net2w), d_net2b)
            #d_net2 = tf.nn.dropout(tf.nn.relu(d_net2l), 0.5)
            d_net2 = tf.nn.relu(d_net2l)
            self.lr_mult[d_net2w] = 10
            self.lr_mult[d_net2b] = 20
        with tf.variable_scope("d_net3"):
            d_net3w = tf.get_variable("w", [DIS_DIM, 1],
                initializer=tf.random_normal_initializer(0, 0.3))
            d_net3b = tf.get_variable("b",
                initializer=tf.zeros([1]))
            d_net3l = tf.nn.bias_add(tf.matmul(d_net2, d_net3w), d_net3b)
            self.lr_mult[d_net3w] = 10
            self.lr_mult[d_net3b] = 20
        self.d_net_list = [d_net1w, d_net1b, d_net2w, d_net2b, d_net3w, d_net3b]
        return d_net3l

    def fc7_residual(self, fc7):
        with tf.variable_scope("res1"):
            res1w = tf.get_variable("w", [self.bottleneck_dim, self.bottleneck_dim],
                initializer=tf.random_normal_initializer(0, 0.01))
            res1b = tf.get_variable("b",
                initializer=tf.zeros([self.bottleneck_dim]))
            res1l = tf.nn.bias_add(tf.matmul(fc8, res1w), res1b)
            res1 = tf.nn.relu(res1l)
            self.lr_mult[res1w] = 10
            self.lr_mult[res1b] = 20
        with tf.variable_scope("res2"):
            res2w = tf.get_variable("w", [self.bottleneck_dim, self.bottleneck_dim],
                initializer=tf.random_normal_initializer(0, 0.01))
            res2b = tf.get_variable("b",
                initializer=tf.zeros([self.bottleneck_dim]))
            res2l = tf.nn.bias_add(tf.matmul(res1, res2w), res2b)
            res2 = tf.add(fc8, res2l)
            self.lr_mult[res2w] = 10
            self.lr_mult[res2b] = 20
        self.res_list = [res1w, res1b, res2w, res2b]
        return res2

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
        max_iter = 50000,
        batch_size = 64,
        learning_rate = 0.001,
        n_class = 10,
        test_size = 265,
        train_size = 958,
        test_iter = 500,
        save_dir = "",
        save_prefix = "",
        model_weights = "./models/imported/caffe/pretrain_model.npy"
        )
    net = Net(config)
    source_set = Dataset("./data/office/amazon_10_list.txt", config['n_class'])
    target_set = Dataset("./data/office/webcam_9_list.txt", config['n_class'])
    net.train(source_set, target_set)

train()
