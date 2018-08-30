
import time
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle as pk

class layer_params:
    def __init__(self):
        self.n_maps= [8,16,32,64,128,256,512]
        self.n_conv = [2,2,2,2,2,2,2,2,2,2]
        self.kernel_size =(3,3)
        self.n_chan = 1
        self.batch_size=1
        self.ip_h = 256
        self.ip_w = 256
        self.num_down_layers=4
        self.reg =0.1
        self.layer_cnt=0
        self.logdir='./logdir'
        self.debug_level=1

def multi_conv(input,prm):

    nxt_layer_ip = input
    conv=[]
    with tf.name_scope("conv_layer_{}".format(prm.layer_cnt)):
        with tf.variable_scope("conv_layer_{}".format(prm.layer_cnt)):
            for conv_layer_cnt in range(0,prm.n_conv[0]): #TBD for n_conv
                tmp= tf.layers.conv2d(nxt_layer_ip,prm.n_maps[0],prm.kernel_size,#TBD for n_maps
                            activation=None,
                            padding='same',
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(prm.reg),
                            #kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                            #bias_initializer=tf.truncated_normal_initializer(stddev=  -0.1, dtype=tf.float32),
                            name =  "conv2d_{}".format(conv_layer_cnt))

                with tf.name_scope("ReLu"):
                    nxt_layer_ip = tf.nn.relu(tmp, name="relu{}".format(conv_layer_cnt))
                conv.append(nxt_layer_ip)
        pool = tf.layers.max_pooling2d(
                conv[-1], (2, 2), strides=(2, 2), name="pool_{}".format(prm.layer_cnt))
        return pool
def up_multi_conv(down_layer,up_layer,prm):
    with tf.name_scope("up_layer_{}".format(prm.layer_cnt)):
        with tf.variable_scope("up_layer_{}".format(prm.layer_cnt)):
            with tf.name_scope("deconv_{}".format(prm.layer_cnt)):
                up_conv_prev_layer = tf.layers.conv2d_transpose(up_layer,prm.n_maps[0],#TBD for n_maps
                                                                kernel_size=2,
                                                                strides=2,
                                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(prm.reg),
                                                                name="upsample_{}".format(prm.layer_cnt)
                                                                )
                if prm.debug_level == 2:
                    up_conv_prev_layer = tf.Print(up_conv_prev_layer, [], "Up layer,    Upsampled,    down_layer,")
                    up_conv_prev_layer = tf.Print(up_conv_prev_layer,
                                                  [tf.shape(up_layer), tf.shape(up_conv_prev_layer), tf.shape(down_layer)])

            with tf.name_scope("concat_{}".format(prm.layer_cnt)):
                concat_layer = tf.concat([up_layer,up_layer],axis=0,name ="concat_".format(prm.layer_cnt))


    up_conv = multi_conv(concat_layer,prm)

    return up_conv_prev_layer


def make_unet_v1(X,prm):
    dims = tf.shape(X)
    down_resol_thresh=4
    next_layer_ip = X
    down_layers=[]
    up_layers = []
    for layer_cnt in range(0,prm.num_down_layers):
        print(" creating layer{}".format(layer_cnt))
        prm.layer_cnt = layer_cnt
        down_layers.append(multi_conv(next_layer_ip,prm))
        next_layer_ip = down_layers[-1]

    for layer_cnt in range(0,prm.num_down_layers-1):
        print(" Up layers",layer_cnt)
        prm.layer_cnt = layer_cnt+prm.num_down_layers
        next_layer_ip = up_multi_conv(down_layers[prm.num_down_layers-(2+layer_cnt)],next_layer_ip, prm)
        up_layers.append(next_layer_ip)
    return down_layers,up_layers

prm = layer_params()
X= tf.placeholder(tf.float32, (prm.batch_size, prm.ip_h, prm.ip_w, prm.n_chan))
down_layers,up_layers = make_unet_v1(X,prm)
train_logdir = os.path.join(prm.logdir, "train",time.strftime("%m_%d_%H_%M_%S"))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)

    test_x = np.zeros((prm.batch_size, prm.ip_h, prm.ip_w, prm.n_chan),dtype=float)
    dims_up = sess.run([down_layers[1],up_layers[2]], feed_dict={ X:test_x})
