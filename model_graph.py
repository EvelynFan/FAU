import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from functools import partial
from config import cfg
from base import ModelDesc
from logger import colorlogger
from basemodel import resnet50, resnet_arg_scope, resnet_v1
resnet_arg_scope = partial(resnet_arg_scope, bn_trainable=cfg.bn_train)

class Model_graph(ModelDesc):
    def head_net(self, blocks, is_training, trainable=True):
        normal_initializer = tf.truncated_normal_initializer(0, 0.01)
        msra_initializer = tf.contrib.layers.variance_scaling_initializer()
        batch_size = blocks[-1].get_shape()[0].value
        k = 7  
        with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
            out = slim.conv2d_transpose(blocks[-1], 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up1')
            out = tf.reshape(out, [batch_size, 256, -1]) 

            out_tmp = tf.squeeze(out)
            if batch_size == 1:
                out_tmp = tf.expand_dims(out_tmp, 0)
            feature_shape = out_tmp.get_shape()
            num_nodes = feature_shape[1].value
            num_dims = feature_shape[2].value
            feature_inner = -2*tf.matmul(out_tmp, tf.transpose(out_tmp, perm=[0, 2, 1]))
            feature_square = tf.reduce_sum(tf.square(out_tmp), axis=-1, keep_dims=True)
            feature_square_T = tf.transpose(feature_square, perm=[0, 2, 1])
            adjacency_matrix = feature_square + feature_inner + feature_square_T
            _, knn_idx = tf.nn.top_k(-adjacency_matrix, k=k)
            feature_central = out_tmp
            idx = tf.reshape(tf.range(batch_size) * num_nodes, [batch_size, 1, 1]) 
            feature_neighbors = tf.gather(tf.reshape(out_tmp, [-1, num_dims]), knn_idx+idx)
            feature_central = tf.tile(tf.expand_dims(feature_central, axis=-2), [1, 1, k, 1])
            edge_feature = tf.concat([feature_central,feature_neighbors-feature_central], axis=-1)

            out = slim.conv2d(edge_feature, 256, [1, 1], stride=1, scope='conv1')
            out = tf.reduce_max(out, axis=-2, keep_dims=True) 
            out = tf.reshape(out, [batch_size,16,16,256])  

            out = slim.conv2d_transpose(out, 128, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up2')

            out = tf.reshape(out, [batch_size, 128, -1]) 

            out_tmp = tf.squeeze(out)
            if batch_size == 1:
                out_tmp = tf.expand_dims(out_tmp, 0)
            feature_shape = out_tmp.get_shape()
            num_nodes = feature_shape[1].value
            num_dims = feature_shape[2].value
            #calculate adjacency matrix
            feature_inner = -2*tf.matmul(out_tmp, tf.transpose(out_tmp, perm=[0, 2, 1]))
            feature_square = tf.reduce_sum(tf.square(out_tmp), axis=-1, keep_dims=True)
            feature_square_T = tf.transpose(feature_square, perm=[0, 2, 1])
            adjacency_matrix = feature_square + feature_inner + feature_square_T
            #k-NN graph
            _, knn_idx = tf.nn.top_k(-adjacency_matrix, k=k)
            feature_central = out_tmp
            idx = tf.reshape(tf.range(batch_size) * num_nodes, [batch_size, 1, 1]) 
            feature_neighbors = tf.gather(tf.reshape(out_tmp, [-1, num_dims]), knn_idx+idx)
            feature_central = tf.tile(tf.expand_dims(feature_central, axis=-2), [1, 1, k, 1])
            edge_feature = tf.concat([feature_central,feature_neighbors-feature_central], axis=-1)

            out = slim.conv2d(edge_feature, 1024, [1, 1], stride=1, scope='conv2')
            out = tf.reduce_max(out, axis=-2, keep_dims=True) 
            out = tf.reshape(out, [batch_size,32,32,128])

            out = slim.conv2d_transpose(out, 64, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up3')

            out = tf.reshape(out, [batch_size, 64, -1]) 

            out_tmp = tf.squeeze(out)
            if batch_size == 1:
                out_tmp = tf.expand_dims(out_tmp, 0)
            feature_shape = out_tmp.get_shape()
            num_nodes = feature_shape[1].value
            num_dims = feature_shape[2].value
            feature_inner = -2*tf.matmul(out_tmp, tf.transpose(out_tmp, perm=[0, 2, 1]))
            feature_square = tf.reduce_sum(tf.square(out_tmp), axis=-1, keep_dims=True)
            feature_square_T = tf.transpose(feature_square, perm=[0, 2, 1])
            adjacency_matrix = feature_square + feature_inner + feature_square_T
            _, knn_idx = tf.nn.top_k(-adjacency_matrix, k=k)
            feature_central = out_tmp
            idx = tf.reshape(tf.range(batch_size) * num_nodes, [batch_size, 1, 1]) 
            feature_neighbors = tf.gather(tf.reshape(out_tmp, [-1, num_dims]), knn_idx+idx)
            feature_central = tf.tile(tf.expand_dims(feature_central, axis=-2), [1, 1, k, 1])
            edge_feature = tf.concat([feature_central,feature_neighbors-feature_central], axis=-1)

            out = slim.conv2d(edge_feature, 4096, [1, 1], stride=1, scope='conv3')
            out = tf.reduce_max(out, axis=-2, keep_dims=True) 
            out = tf.reshape(out, [batch_size,64,64,64])

            out = slim.conv2d(out, cfg.num_AU_points, [1, 1],
                    trainable=trainable, weights_initializer=msra_initializer,
                    padding='SAME', normalizer_fn=None, activation_fn=None,
                    scope='out')
        return out

    def render_gaussian_heatmap(self, coord, output_shape, sigma):     
        x = [i for i in range(output_shape[1])]
        y = [i for i in range(output_shape[0])]
        xx,yy = tf.meshgrid(x,y)
        xx = tf.reshape(tf.to_float(xx), (1,*output_shape,1))
        yy = tf.reshape(tf.to_float(yy), (1,*output_shape,1))          
        x = tf.floor(tf.reshape(coord[:,:,0],[-1,1,1,cfg.num_AU_points]) / cfg.input_shape[1] * output_shape[1] + 0.5)
        y = tf.floor(tf.reshape(coord[:,:,1],[-1,1,1,cfg.num_AU_points]) / cfg.input_shape[0] * output_shape[0] + 0.5)
        heatmap = tf.exp(-(((xx-x)/tf.to_float(sigma))**2)/tf.to_float(2) -(((yy-y)/tf.to_float(sigma))**2)/tf.to_float(2))
        return heatmap * 255.

    def make_network(self,is_train):
        if is_train:
            image = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.input_shape, 3])
            coord = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_AU_points, 2])
            intensity = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_AU_points])
            self.set_inputs(image, coord, intensity)
            backbone = eval(cfg.backbone)
            resnet_fms = backbone(image, is_train, bn_trainable=True)
            heatmap_outs = self.head_net(resnet_fms,is_train)
        else:
            image = tf.placeholder(tf.float32, shape=[cfg.test_batch_size, *cfg.input_shape, 3])
            self.set_inputs(image)
            backbone = eval(cfg.backbone)
            resnet_fms = backbone(image, is_train, bn_trainable=True)
            heatmap_outs = self.head_net(resnet_fms,is_train)
        
        if is_train:           
            intensity_map = tf.reshape(intensity, [cfg.batch_size, 1, 1, cfg.num_AU_points])
            gt_heatmap = tf.stop_gradient(self.render_gaussian_heatmap(coord, cfg.output_shape, cfg.sigma))
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.square(heatmap_outs - gt_heatmap*intensity_map))#MSE Loss       
                tf.summary.scalar('loss',loss)
            self.add_tower_summary('loss', loss)
            self.set_loss(loss)    
        else:     
            self.set_outputs(heatmap_outs)

