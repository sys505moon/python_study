#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 19:02:39 2017

@author: moon
"""

import tensorflow as tf
import os
#import numpy as np
#from PIL import Image


image_dir =  os.getcwd() + "/python/Test_Dataset_png/Face00007.png"
filename_list = [image_dir]

filename_queue = tf.train.string_input_producer(filename_list)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

image_decoded = tf.image.decode_png(value)

image_width = 49
image_height = 61

x = tf.placeholder(tf.float32, shape = [None, image_width, image_height])
y_ = tf.placeholder(tf.float32, shape = [None , 2])

W_hidden1 = tf.Variable(tf.truncated_normal([5,5,1,32]))
b_hidden1 = tf.Variable(tf.zeros([32]))

x_image = tf.reshape(x, [-1, image_width, image_height, 1])

conv1 = tf.nn.conv2d(x_image, W_hidden1, strides = [1,1,1,1], padding = "SAME")
hidden1 = tf.nn.relu(conv1 + b_hidden1)

W_hidden2 = tf.Variable(tf.truncated_normal([5,5,32,64]))
b_hidden2 = tf.Variable(tf.zeros([64]))

conv2 = tf.nn.conv2d(hidden1, W_hidden2, strides = [1,1,1,1], padding="SAME")
hidden2 = tf.nn.relu(conv2 + b_hidden2)

    


with tf.Session() as sess:    
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(coord = coord)
    
    image = sess.run(image_decoded)
    
    coord.request_stop()
    coord.join(thread)
    
#1