#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:43:22 2017

@author: moon
"""

import os
import tensorflow as tf
import numpy as np
import pprint
import functions
os.getcwd()
pp = pprint.PrettyPrinter(indent =4)



sample = "if you want you"
idx2char = list(set(sample))
char2idx = {c : i for i, c in enumerate(idx2char)}

sample_idx = [char2idx[c] for c in sample]


x_data = [sample[:-1]]
y_data = [sample[1:]]

####################################### RNN 12-1

h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

sess = tf.Session()
hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
x_data = np.array([[h,e,l,l,o],
                   [e,o,l,l,l],
                   [l,l,e,e,l]], dtype = np.float32)
print(x_data.shape)
print(x_data)
output, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
sess.run(tf.global_variables_initializer())
print(output.eval())

##################################### RNN Lab 12-2 #############

idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0,1,0,2,3,3]]
x_one_hot = [[[1,0,0,0,0],
              [0,1,0,0,0],
              [1,0,0,0,0],
              [0,0,1,0,0],
              [0,0,0,1,0],
              [0,0,0,1,0]]]
y_data = [[1,0,2,3,3,4]]

num_classes = 5
input_dim = 5
hidden_size = 5
batch_size = 1
sequence_length = 6

x = tf.placeholder(tf.float32,
                   [None, sequence_length, input_dim])
y = tf.placeholder(tf.int32, [None, sequence_length])

cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
initial_state = cell.zero_state(batch_size, tf.float32)
print(initial_state)
functions.showOperation(initial_state)
output, _states = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)

#x_for_fc = tf.reshape(output, [-1, hidden_size])
#output = tf.contrib.layers.fully_connected(inputs=x_for_fc, num_outputs=num_classes, activation_fn=None)

#output = tf.reshape(output, [batch_size, sequence_length, num_classes])

weight = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=output, targets=y, weights=weight)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(output, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict = {x : x_one_hot, y : y_data})
        result = sess.run(prediction, feed_dict= {x : x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true y: ", y_data)
        
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))



for c in sample:
    print(c)
    
    

############################### Lab 12-2 related ###########33
# URL : pythonkim.tistoty.com/

import tensorflow as tf
import numpy as np
import functions

char_rdic = ['h', 'e', 'l', 'o'] # id -> char
char_dic = {w : i for i, w in enumerate(char_rdic)} # char -> id
print (char_dic)

ground_truth = [char_dic[c] for c in 'hello']
print (ground_truth)

x_data = np.array([[1,0,0,0], # h
                   [0,1,0,0], # e
                   [0,0,1,0], # l
                   [0,0,1,0]], # l
                 dtype = 'f')

x_data = tf.one_hot(ground_truth[:-1], len(char_dic), 1.0, 0.0, -1)
print(x_data)
functions.showOperation(x_data)


# Configuration
rnn_size = len(char_dic) # 4
batch_size = 1
output_size = 4



# RNN Model
#rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units = rnn_size,
#                                       input_size = None, # deprecated at tensorflow 0.9
#                                       #activation = tanh,
#                                       reuse = True
#                                       )
rnn_cell = tf.contrib.rnn.BasicRNNCell(rnn_size, reuse = True)
print(rnn_cell)

initial_state = rnn_cell.zero_state(batch_size, tf.float32)
print(initial_state)
functions.showOperation(initial_state)


initial_state_1 = tf.zeros([batch_size, rnn_cell.state_size]) #  위 코드와 같은 결과
print(initial_state_1)

x_split = tf.split(x_data, len(char_dic), 0) # 가로축으로 4개로 split
print(x_split)
functions.showOperation(x_split)
"""
[[1,0,0,0]] # h
[[0,1,0,0]] # e
[[0,0,1,0]] # l
[[0,0,1,0]] # l
"""
#outputs, state = tf.contrib.rnn.static_rnn(cell = rnn_cell, inputs = x_split, initial_state = initial_state)
outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, x_split, initial_state)
print (outputs)
functions.showOperation(outputs)
print (state)
functions.showOperation(state)

logits = tf.reshape(tf.concat(outputs, 1), # shape = 1 x 16
                    [-1, rnn_size])        # shape = 4 x 4
logits.get_shape()
"""
[[logit from 1st output],
[logit from 2nd output],
[logit from 3rd output],
[logit from 4th output]]
"""
targets = tf.reshape(ground_truth[1:], [-1]) # a shape of [-1] flattens into 1-D
targets.get_shape()

weights = tf.ones([len(char_dic) * batch_size])

loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.argmax(logits, 1))
        print(result, [char_rdic[t] for t in result])
        
        
###################################### Lab 12-2 related ###############
# URL : https://github.com/proauto/ML_Practice/blob/master/RNN.py
# reuse error solutions : tf.get_variable_scope().reuse_variables()

import tensorflow as tf
import numpy as np


# 데이터 생성
# char_rdic : Hello World 출력을 위한 알파벳 카테고리 # id -> char
# char_dic : 카테고리를 알파벳에서 숫자로 바꾼다. # char -> id
# x_data : RNN의 입력으로 넣어줄 데이터 - HelloWorl
# sample : 정답인 HelloWorld를 indexing 해준다.
char_rdic = ['H','e','l','o','W','r','d']
char_dic = {w: i for i, w in enumerate(char_rdic)}
x_data = np.array([[1, 0, 0, 0, 0, 0, 0], #H
                   [0, 1, 0, 0, 0, 0, 0], #e
                   [0, 0, 1, 0, 0, 0, 0], #l
                   [0, 0, 1, 0, 0, 0, 0], #l
                   [0, 0, 0, 1, 0, 0, 0], #o
                   [0, 0, 0, 0, 1, 0, 0], #W
                   [0, 0, 0, 1, 0, 0, 0], #o
                   [0, 0, 0, 0, 0, 1, 0], #r
                   [0, 0, 1, 0, 0, 0, 0], #l
                    ],
                  dtype='f')
sample = [char_dic[c] for c in "HelloWorld"]  # to index


# 기본 설정
# rnn_size : RNN 입출력의 크기
# time_step_size : 출력을 반복하는 횟수 ('HelloWorl' -> predict 'elloWorld')
rnn_size = len(char_dic)
time_step_size = 9
learning_rate = 0.03


# RNN model
# rnn_cell : RNN 셀
# state : hidden state 초기화
# X_split : 입력의 크기
# ouputs, state : 출력과 갱신된 hidden state
rnn_cell = tf.contrib.rnn.BasicRNNCell(rnn_size, reuse = True)
state = tf.zeros([1, rnn_cell.state_size])
X_split = tf.split(x_data, time_step_size, 0)
outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, X_split, initial_state = state)


# seq2seq.py를 이용하기 위한 초기화
# logits : list of 2D Tensors of shape [batch_size x num_decoder_symbols].
# targets : list of 1D batch-sized int32 Tensors of the same length as logits
# weights : list of 1D batch-sized float-Tensors of the same length as logits
logits = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])
targets = tf.reshape(sample[1:], [-1])
weights = tf.ones([time_step_size * 1])


# cost : Tensorflow에서는 sequence_loss_by_example를 기본으로 제공한다
# train_op : adam optimizer로 오차 최소화
cost = tf.reduce_sum(tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[targets],[weights]))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Tensorflow 세션 실행
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 100 번 학습
for i in range(100):
    sess.run(train_op)
    result = sess.run(tf.arg_max(logits, 1))

    output = ''.join([char_rdic[t] for t in result])
    print(i+1, "/ 100 ","H" + output)
    


################## RNN Lab 12-3 #######################
# reuse error solutions : tf.get_variable_scope().reuse_variables()

import tensorflow as tf
import numpy as np

sample = " if you want you"
idx2char = list(set(sample))
char2idx = {c : i for i, c in enumerate(idx2char)}

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]] # double list ?
y_data = [sample_idx[1:]]

# hyper parameters
dic_size = len(char2idx)
rnn_hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
sequence_length = len(sample)-1

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

x_one_hot = tf.one_hot(X, num_classes)

cell = tf.contrib.rnn.BasicLSTMCell(num_units = rnn_hidden_size,
                                    state_is_tuple = True,
                                    reuse = True)
initial_state = cell.zero_state(batch_size, tf.float32)

outputs, _state = tf.nn.dynamic_rnn(cell, x_one_hot,
                                    initial_state = initial_state,
                                    dtype = tf.float32)
#outputs, _state = tf.contrib.rnn.static_rnn(cell, x_one_hot,
#                                            initial_state = initial_state,
#                                            dtype = tf.float32)
weight = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs,
                                                 targets = Y,
                                                 weights = weight)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        l, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction:", ''.join(result_str))


############## Lap 12-3 Long Sentence : practice version ##################

# reuse error solutions : tf.get_variable_scope().reuse_variables()

import tensorflow as tf
import numpy as np

sentence = ("if you want to build a ship, don't drum up people together to"
            "collect wood and don't assign them tasks and work, but rather"
            "teach them to long for the endless immensity of the sea.")
char_set = list(set(sentence))
char_dic = {w:i for i, w in enumerate(char_set)}

# hyper parameters
data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
seq_length = 10 # Any arbitrary number

dataX = []
dataY = []

for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i+1 : i+seq_length+1]
    print(i, x_str, '->', y_str)
    
    x = [char_dic[c] for c in x_str] # make index 
    y = [char_dic[c] for c in y_str]
    
    dataX.append(x)
    dataY.append(y)
    
batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None, seq_length])
Y = tf.placeholder(tf.int32, [None, seq_length])

x_one_hot = tf.one_hot(X, num_classes)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
                                    state_is_tuple = True,
                                    reuse = True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _state = tf.nn.dynamic_rnn(cell, x_one_hot,
                                    initial_state=initial_state,
                                    dtype = tf.float32)

weight = tf.ones([batch_size, seq_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,
                                                 targets=Y,
                                                 weights=weight)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis = 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        l, _ = sess.run([loss, train], feed_dict={X:dataX, Y:dataY})
        result = sess.run(prediction, feed_dict={X:dataX})
        
        result_str = [char_set[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction:", ''.join(result_str))

############## Lap 12-3 Long Sentence : Solution version ##################
# reuse error solutions : tf.get_variable_scope().reuse_variables()

import tensorflow as tf
import numpy as np

sentence = ("if you want to build a ship, don't drum up people together to"
            "collect wood and don't assign them tasks and work, but rather"
            "teach them to long for the endless immensity of the sea.")
char_set = list(set(sentence))
char_dic = {w:i for i, w in enumerate(char_set)}

# hyper parameters
data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
seq_length = 10 # Any arbitrary number

dataX = []
dataY = []

for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i+1 : i+seq_length+1]
    print(i, x_str, '->', y_str)
    
    x = [char_dic[c] for c in x_str] # make index 
    y = [char_dic[c] for c in y_str]
    
    dataX.append(x)
    dataY.append(y)
    
batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None, seq_length])
Y = tf.placeholder(tf.int32, [None, seq_length])

x_one_hot = tf.one_hot(X, num_classes)

cell = tf.contrib.rnn.MultiRNNCell([cell]*2,
                                    state_is_tuple = True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _state = tf.nn.dynamic_rnn(cell, x_one_hot,
                                    initial_state=initial_state,
                                    dtype = tf.float32)

# additional code
x_for_softmax = tf.reshape(outputs, [-1, hidden_size])

softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(x_for_softmax, softmax_w) + softmax_b
outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])


weight = tf.ones([batch_size, seq_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,
                                                 targets=Y,
                                                 weights=weight)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run([train_op, mean_loss, outputs],
                             feed_dict = {X:x_data, Y:y_data})
    
    for j, result in enumerate(results):
        index = np.argmax(result, axis = 1)
        print(i, j, ''.join([char_set[t] for t in index]), l)











