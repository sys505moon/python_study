#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 20:44:22 2017

@author: moon
"""

import tensorflow as tf

filename_queue = tf.train.string_input_producer(['./python/test.csv'])
num_record = 3
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[1], [1], [1], [1], [1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5, col6, col7, col8, col9 = tf.decode_csv(value, record_defaults)
features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    
    for i in range(num_record):
        example, label = sess.run([features, col9])
        print(" example : ", example)
        print(" label   : ", label)
      
    coord.request_stop()
    coord.join(threads)
    
    