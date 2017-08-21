##################################################################################
# 2016 9.22 Created by LiuShichen                                                #
# Learning to Ranking Hash on Deep Net (Tensorflow version)                      #
#                                                                                #
#                                                                                #
##################################################################################

import os
import sys
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
from datetime import datetime
from math import ceil
import random
from util import ProgressBar, Dataset, MAPs, MAPs_CQ
from sklearn.cluster import MiniBatchKMeans

from nets import inception
import tensorflow.contrib.slim as slim

class DRH(object):
    def __init__(self, config):
        ### Initialize setting
        print ("initializing")
        np.set_printoptions(precision=4)
        self.stage = config['stage']
        self.device = config['device']
        self.output_dim = config['output_dim']
        self.n_class = config['label_dim']
        
        self.subspace_num = config['n_subspace']
        self.subcenter_num = config['n_subcenter']
        self.code_batch_size = config['code_batch_size']
        self.cq_lambda = config['cq_lambda']
        self.max_iter_update_Cb = config['max_iter_update_Cb']
        self.max_iter_update_b = config['max_iter_update_b']
        #self.centers_device = config['centers_device']
        
        self.alpha = config['alpha']
        self.batch_size = config['batch_size']
        self.max_iter = config['max_iter']
        self.img_model = config['img_model']
        self.loss_type = config['loss_type']
        self.console_log = (config['console_log'] == 1)
        self.learning_rate = config['learning_rate']
        self.learning_rate_decay_factor = config['learning_rate_decay_factor']
        self.decay_step = config['decay_step']

        self.margin_param = config['margin_param']
        self.wordvec_dict = config['wordvec_dict']
        self.part_ids_dict = config['part_ids_dict']
        self.partlabel = config['partlabel']
        ### Format as 'path/to/save/dir/lr_{$0}_output_dim{$1}_iter_{$2}'
        #self.save_dir = config['save_dir'] + 'lr_' + str(self.learning_rate) + '_output_' + str(self.output_dim) + '_iter_' + str(self.max_iter)
        self.save_dir = config['save_dir'] + self.loss_type + '-marg-' + str(self.margin_param) + '-' + self.img_model + '_lr_' + str(self.learning_rate) + '_cqlambda_'+ str(self.cq_lambda) + '_subspace_' + str(self.subspace_num) + '_iter_' + str(self.max_iter) + '_output_' + str(self.output_dim) + '_'

        ### Setup session
        print ("launching session")
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        self.sess = tf.Session(config=configProto)

        ### Create variables and placeholders
        
        with tf.device(self.device):
            self.img = tf.placeholder(tf.float32, [self.batch_size, 256, 256, 3])
            self.img_label = tf.placeholder(tf.float32, [self.batch_size, self.n_class])

            self.img_last_layer, self.C = \
                self.load_model(config['model_weights'])
                
            ### Centers shared in different modalities (image & text)
            ### Binary codes for different modalities (image & text)
            self.img_output_all = tf.placeholder(tf.float32, [None, self.output_dim])
            self.img_b_all = tf.placeholder(tf.float32, [None, self.subspace_num * self.subcenter_num])

            self.b_img = tf.placeholder(tf.float32, [None, self.subspace_num * self.subcenter_num])
            self.ICM_m = tf.placeholder(tf.int32, [])
            self.ICM_b_m = tf.placeholder(tf.float32, [None, self.subcenter_num])
            self.ICM_b_all = tf.placeholder(tf.float32, [None, self.subcenter_num * self.subspace_num])
            self.ICM_X = tf.placeholder(tf.float32, [self.code_batch_size, self.output_dim])
            self.ICM_C_m = tf.slice(self.C, [self.ICM_m * self.subcenter_num, 0], [self.subcenter_num, self.output_dim])
            self.ICM_X_residual = tf.add(tf.sub(self.ICM_X, tf.matmul(self.ICM_b_all, self.C)), tf.matmul(self.ICM_b_m, self.ICM_C_m))
            ICM_X_expand = tf.expand_dims(self.ICM_X_residual, 2)
            ICM_C_m_expand = tf.expand_dims(tf.transpose(self.ICM_C_m), 0)
            ICM_sum_squares = tf.reduce_sum(tf.square(tf.squeeze(tf.sub(ICM_X_expand, ICM_C_m_expand))), reduction_indices = 1)
            ICM_best_centers = tf.argmin(ICM_sum_squares, 1)
            self.ICM_best_centers_one_hot = tf.one_hot(ICM_best_centers, self.subcenter_num, dtype = tf.float32)

            self.global_step = tf.Variable(0, trainable=False)
            self.train_op = self.apply_loss_function(self.global_step)
            self.sess.run(tf.initialize_all_variables())
        return

    def load_model(self, img_model_weights):
        if self.img_model == 'alexnet':
            img_output = self.img_alexnet_layers(img_model_weights)
        elif self.img_model == 'vgg':
            img_output = self.img_vgg_layers(img_model_weights)
        elif self.img_model == 'Inception-V4':
            img_output = self.img_incetion_v4_layers(img_model_weights)
        else:
            raise Exception('cannot use such CNN model as ' + self.img_model)
        return img_output

    def img_incetion_v4_layers(self, model_weights):
        self.deep_param_img = {}
        self.train_layers = []
        self.train_last_layer = []
        print ("loading img model")
        
        IMAGE_SIZE = 299
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        distorted_image = tf.image.resize_images(tf.cast(self.img, tf.float32), [height, width])
        
        #distorted_image = tf.reshape(reshaped_image,[self.batch_size * 256 * 256 , 3])
        
        ### Randomly crop a [height, width] section of each image
        #distorted_image = tf.pack([tf.random_crop(tf.image.random_flip_left_right(each_image), [height, width, 3]) for each_image in tf.unpack(reshaped_image)])

        ### Zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img-mean')
            distorted_image = distorted_image - mean
        
        with slim.arg_scope(inception.inception_v4_arg_scope()):
            logits, _ = inception.inception_v4(distorted_image, num_classes=self.output_dim, is_training=True, create_aux_logits=False)
            self.fc8 = tf.nn.tanh(logits)
            
        self.centers = tf.Variable(tf.random_uniform([self.subspace_num * self.subcenter_num, self.output_dim],
                minval = -1, maxval = 1, dtype = tf.float32, name = 'centers'))
        
        checkpoint_exclude_scopes=["InceptionV4/Logits", "InceptionV4/AuxLogits"]
    
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
                    
            if not excluded:
                variables_to_restore.append(var)
            else:
                self.train_last_layer.append(var)
            self.deep_param_img[var.op.name] = var

        self.train_layers = variables_to_restore
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(self.sess, "./inception_v4.ckpt")
        
        self.deep_param_img['C'] = self.centers
        return self.fc8, self.centers

    def img_alexnet_layers(self, model_weights):
        self.deep_param_img = {}
        self.train_layers = []
        self.train_last_layer = []
        print ("loading img model")
        net_data = np.load(model_weights).item()
        
        # swap(2,1,0)
        reshaped_image = tf.cast(self.img, tf.float32)
        tm = tf.Variable([[0,0,1],[0,1,0],[1,0,0]],dtype=tf.float32)
        reshaped_image = tf.reshape(reshaped_image,[self.batch_size * 256 * 256, 3])
        reshaped_image = tf.matmul(reshaped_image,tm)
        reshaped_image = tf.reshape(reshaped_image,[self.batch_size, 256 , 256, 3])
        
        IMAGE_SIZE = 227
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        ### Randomly crop a [height, width] section of each image
        distorted_image = tf.pack([tf.random_crop(tf.image.random_flip_left_right(each_image), [height, width, 3]) for each_image in tf.unpack(reshaped_image)])

        ### Zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img-mean')
            distorted_image = distorted_image - mean

        ### Conv1
        ### Output 96, kernel 11, stride 4
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(net_data['conv1'][0], name='weights')
            conv = tf.nn.conv2d(distorted_image, kernel, [1, 4, 4, 1], padding='VALID')
            biases = tf.Variable(net_data['conv1'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv1'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Pool1
        self.pool1 = tf.nn.max_pool(self.conv1,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
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
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(net_data['conv2'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.lrn1)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(3, output_groups)
            
            biases = tf.Variable(net_data['conv2'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv2'] = [kernel, biases]
            self.train_layers += [kernel, biases]

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
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(net_data['conv3'][0], name='weights')
            conv = tf.nn.conv2d(self.lrn2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv3'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv3'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Conv4
        ### Output 384, pad 1, kernel 3, group 2
        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(net_data['conv4'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.conv3)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.Variable(net_data['conv4'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv4'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Conv5
        ### Output 256, pad 1, kernel 3, group 2
        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(net_data['conv5'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.conv4)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.Variable(net_data['conv5'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv5'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Pool5
        self.pool5 = tf.nn.max_pool(self.conv5,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool5')

        ### FC6
        ### Output 4096
        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc6w = tf.Variable(net_data['fc6'][0], name='weights')
            fc6b = tf.Variable(net_data['fc6'][1], name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            self.fc5 = pool5_flat
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            self.fc6 = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)
            self.fc6o = tf.nn.relu(fc6l)
            self.deep_param_img['fc6'] = [fc6w, fc6b]
            self.train_layers += [fc6w, fc6b]

        ### FC7
        ### Output 4096
        with tf.name_scope('fc7') as scope:
            fc7w = tf.Variable(net_data['fc7'][0], name='weights')
            fc7b = tf.Variable(net_data['fc7'][1], name='biases')
            fc7l = tf.nn.bias_add(tf.matmul(self.fc6, fc7w), fc7b)
            self.fc7 = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)
            fc7lo = tf.nn.bias_add(tf.matmul(self.fc6o, fc7w), fc7b)            
            self.fc7o = tf.nn.relu(fc7lo)
            self.deep_param_img['fc7'] = [fc7w, fc7b]
            self.train_layers += [fc7w, fc7b]

        ### FC8
        ### Output output_dim
        with tf.name_scope('fc8') as scope:
            ### Differ train and val stage by 'fc8' as key
            if 'fc8' in net_data:
                fc8w = tf.Variable(net_data['fc8'][0], name='weights')
                fc8b = tf.Variable(net_data['fc8'][1], name='biases')
            else:
                fc8w = tf.Variable(tf.random_normal([4096, self.output_dim],
                                                       dtype=tf.float32,
                                                       stddev=1e-2), name='weights')
                fc8b = tf.Variable(tf.constant(0.0, shape=[self.output_dim],
                                               dtype=tf.float32), name='biases')
            fc8l = tf.nn.bias_add(tf.matmul(self.fc7, fc8w), fc8b)
            self.fc8 = tf.nn.tanh(fc8l)
            fc8lo = tf.nn.bias_add(tf.matmul(self.fc7o, fc8w), fc8b)
            self.fc8o = tf.nn.tanh(fc8lo)
            self.deep_param_img['fc8'] = [fc8w, fc8b]
            self.train_last_layer += [fc8w, fc8b]
            self.test3 = fc8w
            self.test1 = fc8b
        
        ### load centers
        if 'C' in net_data:
            self.centers = tf.Variable(net_data['C'], name='weights')
        else:
            self.centers = tf.Variable(tf.random_uniform([self.subspace_num * self.subcenter_num, self.output_dim],
                                    minval = -1, maxval = 1, dtype = tf.float32, name = 'centers'))

        self.deep_param_img['C'] = self.centers

        print("img modal loading finished")
        ### Return outputs
        return self.fc8, self.centers

    def img_vgg_layers(self, model_weights):
        self.deep_param_img = {}
        self.train_layers = []
        self.train_last_layer = []
        print ("loading img model")
        net_data = np.load(model_weights)
        
        # swap(2,1,0)
        reshaped_image = tf.cast(self.img, tf.float32)
        tm = tf.Variable([[0,0,1],[0,1,0],[1,0,0]],dtype=tf.float32)
        reshaped_image = tf.reshape(reshaped_image,[self.batch_size * 256 * 256, 3])
        reshaped_image = tf.matmul(reshaped_image,tm)
        reshaped_image = tf.reshape(reshaped_image,[self.batch_size, 256 , 256, 3])
        
        IMAGE_SIZE = 224
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        ### Randomly crop a [height, width] section of each image
        distorted_image = tf.pack([tf.random_crop(tf.image.random_flip_left_right(each_image), [height, width, 3]) for each_image in tf.unpack(reshaped_image)])

        ### Zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img-mean')
            distorted_image = distorted_image - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(net_data['conv1_1_W'], name='weights')
            conv = tf.nn.conv2d(distorted_image, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv1_1_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv1_1_W'] = kernel
            self.deep_param_img['conv1_1_b'] = biases
            self.train_layers += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(net_data['conv1_2_W'], name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv1_2_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv1_2_W'] = kernel
            self.deep_param_img['conv1_2_b'] = biases
            self.train_layers += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(net_data['conv2_1_W'], name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv2_1_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)

            self.deep_param_img['conv2_1_W'] = kernel
            self.deep_param_img['conv2_1_b'] = biases
            self.train_layers += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(net_data['conv2_2_W'], name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv2_2_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv2_2_W'] = kernel
            self.deep_param_img['conv2_2_b'] = biases
            self.train_layers += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(net_data['conv3_1_W'], name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv3_1_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv3_1_W'] = kernel
            self.deep_param_img['conv3_1_b'] = biases
            self.train_layers += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(net_data['conv3_2_W'], name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv3_2_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv3_2_W'] = kernel
            self.deep_param_img['conv3_2_b'] = biases
            self.train_layers += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(net_data['conv3_3_W'], name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv3_3_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv3_3_W'] = kernel
            self.deep_param_img['conv3_3_b'] = biases
            self.train_layers += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(net_data['conv4_1_W'], name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv4_1_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv4_1_W'] = kernel
            self.deep_param_img['conv4_1_b'] = biases
            self.train_layers += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(net_data['conv4_2_W'], name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv4_2_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv4_2_W'] = kernel
            self.deep_param_img['conv4_2_b'] = biases
            self.train_layers += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(net_data['conv4_3_W'], name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv4_3_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv4_3_W'] = kernel
            self.deep_param_img['conv4_3_b'] = biases
            self.train_layers += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(net_data['conv5_1_W'], name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv5_1_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv5_1_W'] = kernel
            self.deep_param_img['conv5_1_b'] = biases
            self.train_layers += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(net_data['conv5_2_W'], name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv5_2_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv5_2_W'] = kernel
            self.deep_param_img['conv5_2_b'] = biases
            self.train_layers += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(net_data['conv5_3_W'], name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv5_3_b'], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv5_3_W'] = kernel
            self.deep_param_img['conv5_3_b'] = biases
            self.train_layers += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # fc6
        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc6w = tf.Variable(net_data['fc6_W'], name='weights')
            fc6b = tf.Variable(net_data['fc6_b'], name='biases')
            
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)

            self.fc6 = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)
            self.deep_param_img['fc6_W'] = fc6w
            self.deep_param_img['fc6_b'] = fc6b
            self.train_layers += [fc6w, fc6b]

        # fc7
        with tf.name_scope('fc7') as scope:
            fc7w = tf.Variable(net_data['fc7_W'], name='weights')
            fc7b = tf.Variable(net_data['fc7_b'], name='biases')
            fc7l = tf.nn.bias_add(tf.matmul(self.fc6, fc7w), fc7b)
            self.fc7 = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)
            
            self.deep_param_img['fc7_W'] = fc7w
            self.deep_param_img['fc7_b'] = fc7b

            self.train_layers += [fc7w, fc7b]

        # fc8
        with tf.name_scope('fc8') as scope:
            if 'fc8_newW' in net_data:
                fc8w = tf.Variable(net_data['fc8_newW'], name='weights')
                fc8b = tf.Variable(net_data['fc8_newb'], name='biases')
            else:
                fc8w = tf.Variable(tf.random_normal([4096, self.output_dim],
                                                       dtype=tf.float32,
                                                       stddev=1e-2), name='weights')
                fc8b = tf.Variable(tf.constant(1.0, shape=[self.output_dim],
                                               dtype=tf.float32), name='biases')

            fc8l = tf.nn.bias_add(tf.matmul(self.fc7, fc8w), fc8b)
            self.fc8 = tf.nn.tanh(fc8l, name='image_last_layer')
            
            self.deep_param_img['fc8_newW'] = fc8w
            self.deep_param_img['fc8_newb'] = fc8b
            self.train_last_layer += [fc8w, fc8b]

        ### load centers
        if 'C' in net_data:
            self.centers = tf.Variable(net_data['C'], name='weights')
        else:
            self.centers = tf.Variable(tf.random_uniform([self.subspace_num * self.subcenter_num, self.output_dim],
                                    minval = -1, maxval = 1, dtype = tf.float32, name = 'centers'))

        self.deep_param_img['C'] = self.centers

        print("img modal loading finished")
        ### Return outputs
        return self.fc8, self.centers

    def save_model(self, model_file=None):
        if model_file == None:
            model_file = self.save_dir
        model = {}
        for layer in self.deep_param_img:
            model[layer] = self.sess.run(self.deep_param_img[layer])
        print ("saving model to %s" % model_file)
        np.save(model_file, np.array(model))
        return

    def apply_loss_function(self, global_step):
        ### loss function
        if self.loss_type == 'ip':
            InnerProduct = tf.mul(self.alpha, tf.matmul(self.img_last_layer, tf.transpose(self.img_last_layer)))
            Sim = tf.clip_by_value(tf.matmul(self.img_label, tf.transpose(self.img_label)), 0.0, 1.0)
            batch_n = tf.shape(self.img_last_layer)[0]
            t_ones = tf.ones([batch_n, batch_n])
            T_cross = tf.sub(tf.log(tf.add(t_ones, tf.exp(InnerProduct))), tf.mul(Sim, InnerProduct))
            E_cross = tf.mul(InnerProduct, tf.sub(1.0, Sim))
            condition_p = tf.less(tf.abs(InnerProduct), 15.0)
            self.cross_entropy_loss = tf.reduce_mean(tf.select(condition_p, T_cross, E_cross))
            self.test1 = Sim
            self.test2 = InnerProduct
            self.test3 = self.img_last_layer
            one_bias = tf.sub(tf.abs(self.img_last_layer), 1.0)
            T = tf.log(tf.div(tf.add(tf.exp(one_bias), tf.exp(-one_bias)), 2.0))
            E = tf.sub(one_bias, tf.log(2.0))
            condition_q = tf.less(one_bias, 15.0)
            q_matrix = tf.select(condition_q, T, E)
            self.q_loss = tf.reduce_mean(q_matrix)
            self.cos_loss = tf.add(self.cross_entropy_loss, tf.mul(self.q_loss, 0.1))
        elif self.loss_type == 'cos':
            # let sim = {0, 1} to be {-1, 1}
            Sim_1 = tf.clip_by_value(tf.matmul(self.img_label, tf.transpose(self.img_label)), 0.0, 1.0)
            Sim_2 = tf.add(Sim_1,tf.constant(-0.5))
            Sim = tf.mul(Sim_2,tf.constant(2.0))
            
            # compute balance param = num of 0 / num of 1
            sum_1 = tf.reduce_sum(Sim_1)
            sum_0 = tf.reduce_sum(tf.abs(Sim))
            balance_param = tf.add(tf.abs(tf.add(Sim_1,tf.constant(-1.0))), tf.mul(tf.div(sum_0, sum_1), Sim_1))
            
            # stop gradient of norm
            const_img = tf.stop_gradient(self.img_last_layer)

            ip_1 = tf.matmul(self.img_last_layer, self.img_last_layer, transpose_b=True)
            def reduce_shaper(t):
                return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])
            #mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(const_img)), reduce_shaper(tf.square(const_img)), transpose_b=True))
            mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(self.img_last_layer)), reduce_shaper(tf.square(self.img_last_layer)), transpose_b=True))
            cos_1 = tf.div(ip_1, mod_1)
            self.cos_loss = tf.reduce_mean(tf.mul(balance_param, tf.square(tf.sub(Sim, cos_1))))

        elif self.loss_type == 'cos-nostopgradient':
            # let sim = {0, 1} to be {-1, 1}
            Sim_1 = tf.clip_by_value(tf.matmul(self.img_label, tf.transpose(self.img_label)), 0.0, 1.0)
            Sim_2 = tf.add(Sim_1,tf.constant(-0.5))
            Sim = tf.mul(Sim_2,tf.constant(2.0))
            
            # compute balance param = num of 0 / num of 1
            sum_1 = tf.reduce_sum(Sim_1)
            sum_0 = tf.reduce_sum(tf.abs(Sim))
            balance_param = tf.add(tf.abs(tf.add(Sim_1,tf.constant(-1.0))), tf.mul(tf.div(sum_0, sum_1), Sim_1))
            
            # stop gradient of norm
            const_img = tf.stop_gradient(self.img_last_layer)

            ip_1 = tf.matmul(self.img_last_layer, self.img_last_layer, transpose_b=True)
            def reduce_shaper(t):
                return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])
            #mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(const_img)), reduce_shaper(tf.square(const_img)), transpose_b=True))
            mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(self.img_last_layer)), reduce_shaper(tf.square(self.img_last_layer)), transpose_b=True))
            cos_1 = tf.div(ip_1, mod_1)
            self.cos_loss = tf.reduce_mean(tf.mul(balance_param, tf.square(tf.sub(Sim, cos_1))))
        
        elif self.loss_type == 'cos-delta':
            # let sim = {0, 1} to be {-1, 1}
            Sim_1 = tf.clip_by_value(tf.matmul(self.img_label, tf.transpose(self.img_label)), 0.0, 1.0)
            Sim_t = tf.add(Sim_1,tf.constant(-0.5))
            Sim = tf.mul(Sim_t,tf.constant(2.0))

            # compute sim_0: if sij = 0, Sim_0 ij = 1; else Sim_0 ij = 0
            Sim_0 = tf.abs(tf.add(Sim_1,tf.constant(-1.0)))
            
            # compute balance param = num of 0 / num of 1
            sum_1 = tf.reduce_sum(Sim_1)
            sum_0 = tf.reduce_sum(tf.abs(Sim))
            balance_param = tf.add(Sim_0, tf.mul(tf.div(sum_0, sum_1), Sim_1))
            
            # stop gradient of norm
            const_img = tf.stop_gradient(self.img_last_layer)

            ip_1 = tf.matmul(self.img_last_layer, self.img_last_layer, transpose_b=True)
            def reduce_shaper(t):
                return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])
            #mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(const_img)), reduce_shaper(tf.square(const_img)), transpose_b=True))
            mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(self.img_last_layer)), reduce_shaper(tf.square(self.img_last_layer)), transpose_b=True))
            cos_1 = tf.div(ip_1, mod_1)
            
            # compute delta
            sij_cos = tf.sub(Sim, cos_1)
            #delta = tf.add(tf.reduce_mean(tf.mul(Sim_0, sij_cos)), tf.reduce_mean(tf.mul(Sim_1, sij_cos)))
            delta = tf.constant(self.margin_param)
            
            sij_cos_delta = tf.sub(tf.square(sij_cos), tf.square(delta))
            condition = tf.less(sij_cos_delta, tf.constant(0.0))
            final = tf.select(condition, tf.constant(0.0,shape=[self.batch_size, self.batch_size]), sij_cos_delta)
            self.cos_loss = tf.reduce_mean(tf.mul(balance_param, final))
        
        elif self.loss_type == 'cos-delta-1':
            # let sim = {0, 1} to be {-1, 1}
            Sim_1 = tf.clip_by_value(tf.matmul(self.img_label, tf.transpose(self.img_label)), 0.0, 1.0)
            Sim_t = tf.add(Sim_1,tf.constant(-0.5))
            Sim = tf.mul(Sim_t,tf.constant(2.0))

            # compute sim_0: if sij = 0, Sim_0 ij = 1; else Sim_0 ij = 0
            Sim_0 = tf.abs(tf.add(Sim_1,tf.constant(-1.0)))
            
            # compute balance param = num of 0 / num of 1
            sum_1 = tf.reduce_sum(Sim_1)
            sum_0 = tf.reduce_sum(tf.abs(Sim))
            balance_param = tf.add(Sim_0, tf.mul(tf.div(sum_0, sum_1), Sim_1))
            
            # stop gradient of norm
            const_img = tf.stop_gradient(self.img_last_layer)

            ip_1 = tf.matmul(self.img_last_layer, self.img_last_layer, transpose_b=True)
            def reduce_shaper(t):
                return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])
            #mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(const_img)), reduce_shaper(tf.square(const_img)), transpose_b=True))
            mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(self.img_last_layer)), reduce_shaper(tf.square(self.img_last_layer)), transpose_b=True))
            cos_1 = tf.div(ip_1, mod_1)
            
            # compute delta
            delta = tf.constant(self.margin_param)
            
            sij_cos_delta = tf.sub(delta, tf.mul(Sim, cos_1))
            #sij_cos_delta = tf.sub(tf.square(sij_cos), tf.square(delta))
            condition = tf.less(sij_cos_delta, tf.constant(0.0))
            final = tf.select(condition, tf.constant(0.0,shape=[self.batch_size, self.batch_size]), sij_cos_delta)
            self.cos_loss = tf.reduce_mean(tf.mul(balance_param, tf.square(final)))
        
        elif self.loss_type == 'cos-semi':
            def reduce_shaper(t):
                return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])
            # let sim = {0, 1} to be {-1, 1}
            Sim_1 = tf.clip_by_value(tf.matmul(self.img_label, tf.transpose(self.img_label)), 0.0, 1.0)
            Sim_t = tf.add(Sim_1,tf.constant(-0.5))
            Sim = tf.mul(Sim_t,tf.constant(2.0))

            # compute sim_0: if sij = 0, Sim_0 ij = 1; else Sim_0 ij = 0
            Sim_0 = tf.abs(tf.add(Sim_1,tf.constant(-1.0)))
            
            # (semi) supervised indicator: if both have label, then sup_ind == 1, else sup_ind = 0
            sup_ind_t = tf.clip_by_value(reduce_shaper(self.img_label), 0.0, 1.0)
            sup_ind = tf.matmul(sup_ind_t, tf.transpose(sup_ind_t))
            semi_sup_ind = tf.abs(tf.add(sup_ind,tf.constant(-1.0)))
            
            # compute balance param = num of 0 / num of 1
            sum_1 = tf.reduce_sum(Sim_1)
            sum_0 = tf.reduce_sum(tf.abs(Sim))
            balance_param = tf.add(Sim_0, tf.mul(tf.div(sum_0, sum_1), Sim_1))
            
            # stop gradient of norm
            const_img = tf.stop_gradient(self.img_last_layer)

            ip_1 = tf.matmul(self.img_last_layer, self.img_last_layer, transpose_b=True)
            #mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(const_img)), reduce_shaper(tf.square(const_img)), transpose_b=True))
            mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(self.img_last_layer)), reduce_shaper(tf.square(self.img_last_layer)), transpose_b=True))
            cos_1 = tf.div(ip_1, mod_1)
            
            # construct new similarity matrix:
            _, indices = tf.nn.top_k(cos_1, k=5)
            semi_sim_t = tf.one_hot(indices, self.batch_size, on_value=1.0, off_value=-1.0, axis=-1)
            semi_sim = tf.reduce_sum(semi_sim_t, 1)
            
            final_sim = tf.add(tf.mul(semi_sim, semi_sup_ind), tf.mul(Sim, sup_ind))
            
            # unsupervised part only have 0.1
            full_ind = tf.add(tf.mul(semi_sup_ind, tf.constant(0.1)), sup_ind)
            
            final_ind = tf.mul(balance_param, full_ind)
            
            # compute delta
            delta = tf.constant(self.margin_param)
            
            sij_cos_delta = tf.sub(delta, tf.mul(final_sim, cos_1))
            condition = tf.less(sij_cos_delta, tf.constant(0.0))
            final = tf.square(tf.select(condition, tf.constant(0.0,shape=[self.batch_size, self.batch_size]), sij_cos_delta))
            self.cos_loss = tf.reduce_mean(tf.mul(final_ind, final))
        
        
        elif self.loss_type == 'cos-nobalance':
            # let sim = {0, 1} to be {-1, 1}
            Sim_1 = tf.clip_by_value(tf.matmul(self.img_label, tf.transpose(self.img_label)), 0.0, 1.0)
            Sim_2 = tf.add(Sim_1,tf.constant(-0.5))
            Sim = tf.mul(Sim_2,tf.constant(2.0))
            
            # compute balance param = num of 0 / num of 1
            sum_1 = tf.reduce_sum(Sim_1)
            sum_0 = tf.reduce_sum(tf.abs(Sim))
            balance_param = tf.add(tf.abs(tf.add(Sim_1,tf.constant(-1.0))), tf.mul(tf.div(sum_0, sum_1), Sim_1))
            
            # stop gradient of norm
            const_img = tf.stop_gradient(self.img_last_layer)

            ip_1 = tf.matmul(self.img_last_layer, self.img_last_layer, transpose_b=True)
            def reduce_shaper(t):
                return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])
            mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(const_img)), reduce_shaper(tf.square(const_img)), transpose_b=True))
            cos_1 = tf.div(ip_1, mod_1)
            self.cos_loss = tf.reduce_mean(tf.square(tf.sub(Sim, cos_1)))

        
        elif self.loss_type == 'cos-w':
            assert self.output_dim == 300

            const_img = tf.stop_gradient(self.img_last_layer)
            
            word_dict = tf.constant(np.loadtxt(self.wordvec_dict), dtype=tf.float32)
            v_label = tf.matmul(self.img_label, word_dict)
            ip_1 = tf.diag_part(tf.matmul(self.img_last_layer, v_label, transpose_b=True))
            #mod_1 = tf.sqrt(tf.mul(tf.reduce_sum(tf.square(self.img_last_layer), 1), tf.reduce_sum(tf.square(v_label), 1)))
            mod_1 = tf.sqrt(tf.mul(tf.reduce_sum(tf.square(const_img), 1), tf.reduce_sum(tf.square(v_label), 1)))
            cos_1 = tf.reshape(tf.div(ip_1, mod_1), [tf.shape(ip_1)[0], 1])
            ip_2 = tf.matmul(self.img_last_layer, word_dict, transpose_b=True)
            def reduce_shaper(t):
                return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])
            #mod_2 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(self.img_last_layer)), reduce_shaper(tf.square(word_dict)), transpose_b=True))
            mod_2 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(const_img)), reduce_shaper(tf.square(word_dict)), transpose_b=True))
            cos_2 = tf.div(ip_2, mod_2)
            #self.cos_loss = -tf.reduce_mean(tf.mul(tf.sub(cos_1, cos_2), tf.abs(tf.sub(cos_1, cos_2))))
            self.cos_loss = -tf.reduce_mean(tf.square(tf.sub(cos_1, cos_2)))
            self.test1 = tf.sub(cos_1, cos_2)
            self.test2 = cos_1
            self.test3 = cos_2
        
        self.cq_loss_img = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(self.img_last_layer, tf.matmul(self.b_img, self.C))), 1))
        self.q_lambda = tf.Variable(self.cq_lambda, name='cq_lambda')
        self.cq_loss = tf.mul(self.q_lambda, self.cq_loss_img)
        self.loss = tf.add(self.cos_loss, self.cq_loss)

        ### Last layer has a 10 times learning rate
        self.lr = tf.train.exponential_decay(self.learning_rate, global_step, self.decay_step, self.learning_rate_decay_factor, staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        grads_and_vars = opt.compute_gradients(self.loss, self.train_layers+self.train_last_layer)
        fcgrad, _ = grads_and_vars[-2]
        fbgrad, _ = grads_and_vars[-1]

        return opt.apply_gradients([(grads_and_vars[i][0], self.train_layers[i]) for i in xrange(len(self.train_layers))]+
                                    [(fcgrad*10, self.train_last_layer[0]),
                                    (fbgrad*20, self.train_last_layer[1])], global_step=global_step)
        #return opt.apply_gradients([(grads_and_vars[0][0], self.train_layers[0]),
                                    #(grads_and_vars[1][0]*2, self.train_layers[1]),
                                    #(grads_and_vars[2][0], self.train_layers[2]),
                                    #(grads_and_vars[3][0]*2, self.train_layers[3]),
                                    #(grads_and_vars[4][0], self.train_layers[4]),
                                    #(grads_and_vars[5][0]*2, self.train_layers[5]),
                                    #(grads_and_vars[6][0], self.train_layers[6]),
                                    #(grads_and_vars[7][0]*2, self.train_layers[7]),
                                    #(grads_and_vars[8][0], self.train_layers[8]),
                                    #(grads_and_vars[9][0]*2, self.train_layers[9]),
                                    #(grads_and_vars[10][0], self.train_layers[10]),
                                    #(grads_and_vars[11][0]*2, self.train_layers[11]),
                                    #(grads_and_vars[12][0], self.train_layers[12]),
                                    #(grads_and_vars[13][0]*2, self.train_layers[13]),
                                    #(fcgrad*10, self.train_last_layer[0]),
                                    #(fbgrad*20, self.train_last_layer[1])], global_step=global_step)
        #return opt.apply_gradients(grads_and_vars[:-2]+[(fcgrad*10, self.train_last_layer[0]), (fbgrad*10, self.train_last_layer[1])], global_step=global_step)

    def initial_centers(self, img_output):
        C_init = np.zeros([self.subspace_num * self.subcenter_num, self.output_dim])
        print "#ZDQ train# initilizing Centers"
        all_output = img_output
        for i in xrange(self.subspace_num):
            kmeans = MiniBatchKMeans(n_clusters=self.subcenter_num).fit(all_output[:, i * self.output_dim / self.subspace_num: (i + 1) * self.output_dim / self.subspace_num])
            C_init[i * self.subcenter_num: (i + 1) * self.subcenter_num, i * self.output_dim / self.subspace_num: (i + 1) * self.output_dim / self.subspace_num] = kmeans.cluster_centers_
            print "step: ", i, " finish"
        return C_init

    def update_centers(self, img_dataset):
        '''
        Optimize:
            self.C = (U * hu^T + V * hv^T) (hu * hu^T + hv * hv^T)^{-1}
            self.C^T = (hu * hu^T + hv * hv^T)^{-1} (hu * U^T + hv * V^T)
            but all the C need to be replace with C^T :
            self.C = (hu * hu^T + hv * hv^T)^{-1} (hu^T * U + hv^T * V)
        '''
        ### cy 8.25 get old C Value
        old_C_value = self.sess.run(self.C)
        
        h = self.img_b_all
        U = self.img_output_all
        smallResidual = tf.constant(np.eye(self.subcenter_num * self.subspace_num, dtype = np.float32) * 0.001)
        Uh = tf.matmul(tf.transpose(h), U)
        hh = tf.add(tf.matmul(tf.transpose(h), h), smallResidual)
        compute_centers = tf.matmul(tf.matrix_inverse(hh), Uh)
        
        update_C = self.C.assign(compute_centers)
        C_value = self.sess.run(update_C, feed_dict = {
            self.img_output_all: img_dataset.output, 
            self.img_b_all: img_dataset.codes,
            })
        
        ### cy 8.25 change all zeros to previous value
        C_sums = np.sum(np.square(C_value), axis=1)
        C_zeros_ids = np.where(C_sums < 1e-8)
        C_value[C_zeros_ids, :] = old_C_value[C_zeros_ids, :]
        self.sess.run(self.C.assign(C_value))
        ### cy 8.25 end

        np.savetxt('log/new_C.txt', C_value)
        
        print 'updated C is:'
        print C_value
        print "non zeros:"
        print len(np.where(np.sum(C_value, 1) != 0)[0])

    def update_codes_ICM(self, output, code):
        '''
        Optimize:
            min || output - self.C * codes ||
            min || output - codes * self.C ||
        args:
            output: [n_train, n_output]
            self.C: [n_subspace * n_subcenter, n_output]
                [C_1, C_2, ... C_M]
            codes: [n_train, n_subspace * n_subcenter]
        '''

        code = np.zeros(code.shape)
        
        for iterate in xrange(self.max_iter_update_b):
#            self.sess.run(tf.initialize_variables([X, X_residual]))
            start = time.time()
            time_init = 0.0
            time_compute_centers = 0.0
            time_append = 0.0
            
            ### cy 8.25 generate random list of subspace sequence
            sub_list = [i for i in range(self.subspace_num)]
            random.shuffle(sub_list)
            for m in sub_list:
            
                best_centers_one_hot_val = self.sess.run(self.ICM_best_centers_one_hot, feed_dict = {
                    self.ICM_b_m: code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num],
                    self.ICM_b_all: code,
                    self.ICM_m: m,
                    self.ICM_X: output,
                })

                code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num] = best_centers_one_hot_val
        return code
    
    def update_codes_batch(self, dataset, batch_size):
        '''
        update codes in batch size
        '''
        total_batch = int(ceil(dataset.n_samples / batch_size))
        print "start update codes in batch size ", batch_size

        dataset.finish_epoch()
        
        for i in xrange(total_batch):
            print "Iter ", i, "of ", total_batch
            output_val, code_val = dataset.next_batch_output_codes(batch_size)
            print output_val, code_val
            codes_val = self.update_codes_ICM(output_val, code_val)
            print np.sum(np.sum(codes_val, 0) != 0)
            dataset.feed_batch_codes(batch_size, codes_val)

        print "update_code wrong:"
        print np.sum(np.sum(dataset.codes, 1) != 4)
        
        print "######### update codes done ##########"

    def train_cq(self, img_dataset):
        print ("%s #train# start training" % datetime.now())
        epoch = 0
        epoch_iter = int(ceil(img_dataset.n_samples / self.batch_size))
        
        for train_iter in xrange(self.max_iter):
            images, labels, codes = img_dataset.next_batch(self.batch_size)
            start_time = time.time()
            
            if epoch > 0:
                assign_lambda = self.q_lambda.assign(self.cq_lambda)
            else:
                assign_lambda = self.q_lambda.assign(0.0)
            self.sess.run([assign_lambda])

            _, cos_loss, cq_loss, lr, output = self.sess.run([self.train_op, self.cos_loss, self.cq_loss, self.lr, self.img_last_layer],
                                    feed_dict={self.img: images,
                                               self.img_label: labels,
                                               self.b_img: codes})
            img_dataset.feed_batch_output(self.batch_size, output)
            duration = time.time() - start_time
#            print test1[:10]
#            print test3[:10, :10]
#            print test2[:10, :10]
#            print test3[:10, :10]
#            print test00
#            print test0
#            print output[:10, :10]
#            print check0
#            print check1
#            print check2
#            print fc8_value
            
            # every epoch: update codes and centers
            if train_iter % (2*epoch_iter) == 0 and train_iter != 0:
                if epoch == 0:
                    with tf.device(self.device):
                        for i in xrange(self.max_iter_update_Cb):
                            print "#ZDQ Train# initialize centers in ", i, " iter"
                            self.sess.run(self.C.assign(self.initial_centers(img_dataset.output)))
                        print "#ZDQ Train# initialize centers done!!!"
                epoch = epoch + 1
                for i in xrange(self.max_iter_update_Cb):
                    print "#ZDQ Train# update codes and centers in ", i, " iter"
                    self.update_codes_batch(img_dataset, self.code_batch_size)
                    self.update_centers(img_dataset)
            
            print("%s #train# step %4d, lr %.8f, cosine margin loss = %.4f, cq loss = %.4f, %.1f sec/batch" % (datetime.now(), train_iter+1, lr, cos_loss, cq_loss, duration))
                
        print ("%s #traing# finish training" % datetime.now())
        self.save_model()
        print ("model saved")

    def train(self, img_dataset):
        print ("%s #train# start training" % datetime.now())
        if self.console_log:
            bar = ProgressBar(total=self.max_iter)
        for train_iter in xrange(self.max_iter):
            images, labels, codes = img_dataset.next_batch(self.batch_size)
            start_time = time.time()

            _, loss, lr = self.sess.run([self.train_op, self.loss, self.lr],
                                    feed_dict={self.img: images,
                                               self.img_label: labels})
            duration = time.time() - start_time
            if self.console_log:
                bar.move("%s #train# step %4d, lr %.8f, loss = %.4f, %.1f sec/batch" % (datetime.now(), train_iter+1, lr, loss, duration))
            else:
                print("%s #train# step %4d, lr %.8f, loss = %.4f, %.1f sec/batch" % (datetime.now(), train_iter+1, lr, loss, duration))
        print ("%s #traing# finish training" % datetime.now())
        self.save_model()
        print ("model saved")

    def validation(self, img_query, img_database, R=100):
        print ("%s #validation# start validation")
        query_batch = int(ceil(img_query.n_samples / self.batch_size))
        print ("%s #validation# totally %d query in %d batches" % (datetime.now(), img_query.n_samples, query_batch))
        if self.console_log:
            bar = ProgressBar(total=query_batch)
        for i in xrange(query_batch):
            images, labels, codes = img_query.next_batch(self.batch_size)

            output, loss = self.sess.run([self.img_last_layer, self.cos_loss],
                                   feed_dict={self.img: images, self.img_label: labels})
            img_query.feed_batch_output(self.batch_size, output)
            if self.console_log:
                bar.move('Cosine Loss: %s'%loss)
            else:
                print('Cosine Loss: %s'%loss)

        database_batch = int(ceil(img_database.n_samples / self.batch_size))
        print ("%s #validation# totally %d database in %d batches" % (datetime.now(), img_database.n_samples, database_batch))
        if self.console_log:
            bar = ProgressBar(total=database_batch)
        for i in xrange(database_batch):
            images, labels, codes = img_database.next_batch(self.batch_size)

            output, loss = self.sess.run([self.img_last_layer, self.cos_loss],
                                   feed_dict={self.img: images, self.img_label: labels})
            img_database.feed_batch_output(self.batch_size, output)
            if self.console_log:
                bar.move('Cosine Loss: %s'%loss)
            else:
                print('Cosine Loss: %s'%loss)

        self.update_codes_batch(img_query, self.code_batch_size)
        self.update_codes_batch(img_database, self.code_batch_size)
        
        print ("%s #validation# calculating MAP@%d" % (datetime.now(), R))
        C_tmp = self.sess.run(self.C)
        mAPs = MAPs_CQ(C_tmp, self.subspace_num, self.subcenter_num, R)
        return {
            'i2i_nocq': mAPs.get_mAPs_by_feature(img_database, img_query),
            'i2i_AQD': mAPs.get_mAPs_AQD(img_database, img_query),
            'i2i_SQD': mAPs.get_mAPs_SQD(img_database, img_query),
        }

def train(train_img, config):
    model = DRH(config)
    img_dataset = Dataset(train_img, config['output_dim'], config['n_subspace'] * config['n_subcenter'])
    model.train_cq(img_dataset)
    return model.save_dir

def validation(database_img, query_img, config):
    model = DRH(config)
    img_database = Dataset(database_img, config['output_dim'], config['n_subspace'] * config['n_subcenter'])
    img_query = Dataset(query_img, config['output_dim'], config['n_subspace'] * config['n_subcenter'])
    model.validation(img_query, img_database, config['R'])
    return

def train_val(train_img, database_img, query_img, config):
    model = DRH(config)
    img_dataset = Dataset(train_img, config['output_dim'], config['n_subspace'] * config['n_subcenter'])
    img_database = Dataset(database_img, config['output_dim'], config['n_subspace'] * config['n_subcenter'])
    img_query = Dataset(query_img, config['output_dim'], config['n_subspace'] * config['n_subcenter'])
    model.train_cq(img_dataset)
    model.validation(img_query, img_database, config['R'])
    return
