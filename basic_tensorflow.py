#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensorflow Lecture in YOUTUBE
"""
import tensorflow as tf

# Leture1 : Data type
 # Constant
const1 = tf.constant([5])
const2 = tf.constant([3])
const3 = tf.constant([2])

const4 = const1 * const2 + const3

sess = tf.Session()
result_const = sess.run(const4)
print(result_const) # Correct value : 17

 #Variable
 
var1 = tf.Variable([5], dtype = tf.float32)
var2 = tf.Variable([3], dtype = tf.float32)
var3 = tf.Variable([2], dtype = tf.float32)

var4 = var1 * var2 + var3

init = tf.global_variables_initializer()
 # In variable, we need to initialize through tf.global_variables_initializer() function
sess.run(init)
result_var = sess.run(var4)
print(result_var) # Correct value : 17

 # Placeholder1 

value1 = 5
value2 = 3
value3 = 2

ph1 = tf.placeholder(dtype = tf.float32)
ph2 = tf.placeholder(dtype = tf.float32)
ph3 = tf.placeholder(dtype = tf.float32)

result_value = value1 * value2 + value3
input_data = {ph1: value1, ph2: value2, ph3: value3}

result_ph = sess.run(result_value, feed_dict = input_data)
print(result_ph) # Correct value : 17

 # Placeholder1
 
image = [1,2,3,4,5]
label = [10,20,30,40,50]

ph_image = tf.placeholder(dtype = tf.float32)
ph_label = tf.placeholder(dtype = tf.float32)

result_tensor = ph_image + ph_label

feed_dict = {ph_image :image, ph_label:label }

result_tensor = sess.run(result_tensor, feed_dict)
print(result_tensor) # Correct valeu : [11,22,33,44,55]

# Leture2 : Design Model

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

input_data = [[1,5,3,7,8,10,12],
              [5,8,10,3,9,7,1]
              ]
label_data = [[0,0,0,1,0],
              [1,0,0,0,0]
              ]

HIDDEN1_SIZE = 10
# hyper paramater
INPUT_SIZE = len(input_data[0]) # 7
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 8
CLASSES = len(label_data[0]) # 5
Learning_Rate = 0.05

x = tf.placeholder(tf.float32, shape = [None, INPUT_SIZE]) # 'None' postions is Batch_size (input_data demension was 2D)
y_ = tf.placeholder(tf.float32, shape = [None, CLASSES]) # label_data demension was 1D, so i don't need write Batch_size value

tensor_map = {x:input_data, y_ : label_data}

 # Input_data - Hidden1 layer
W_h1 = tf.Variable(tf.truncated_normal(shape = [INPUT_SIZE, HIDDEN1_SIZE], dtype = tf.float32)) # weight paramater
b_h1 = tf.Variable(tf.zeros(shape = [HIDDEN1_SIZE]), dtype = tf.float32)

hidden1 = tf.sigmoid(tf.matmul(x, W_h1) + b_h1) # matrix production at input-hidden1 layer

 # Hidden1 layer - Hidden2_layer
W_h2 = tf.Variable(tf.truncated_normal(shape = [HIDDEN1_SIZE, HIDDEN2_SIZE], dtype = tf.float32))
b_h2 = tf.Variable(tf.zeros(shape = [HIDDEN2_SIZE], dtype = tf.float32))

hidden2 = tf.sigmoid(tf.matmul(hidden1, W_h2) + b_h2) # matrix production at hidden1 layer-hidden2 layer

 # Hidden2 layer - label(output)
W_o = tf.Variable(tf.truncated_normal(shape = [HIDDEN2_SIZE, CLASSES], dtype = tf.float32))
b_0 = tf.Variable(tf.zeros(shape = [CLASSES], dtype = tf.float32))

y = tf.sigmoid(tf.matmul(hidden2, W_o) + b_0) # matrix production at hidden2 layer - label(output)

# Training 
 # Csot Function
cost = tf.reduce_mean(-y_*tf.log(y)-(1-y_)*tf.log(1-y)) # if we have y valeu is 1, so we use (1-y)
train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)

 
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    _, loss = sess.run([train, cost], feed_dict = tensor_map)
    if i % 100 == 0:
        print("step: ", i)
        print("loss: ", loss)

# Leture3 : Evaluation & Model Save




























