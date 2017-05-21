#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 18:48:59 2017

@author: moon
"""
import tensorflow as tf
import functions

c1, c2 = tf.constant([3]), tf.constant([1,5])
v1, v2 = tf.Variable([5]), tf.Variable([2,4])

functions.showConstant(c1)
functions.showConstant(c2)
functions.showVariable(v1)
functions.showVariable(v2)

print('---------add---------')
functions.showOperation(tf.add(c1,v1))
functions.showOperation(tf.add(c2,v2))
functions.showOperation(tf.add([c2,v2], [c2,v2]))

print('----------subtract----------')
functions.showOperation(tf.subtract(c1,v1))
functions.showOperation(tf.subtract(c2,v2))

print('----------multiply----------')
functions.showOperation(tf.multiply(c1, v1))
functions.showOperation(tf.multiply(c2, v2))

print('----------div----------')
functions.showOperation(tf.div(c1, v1))
functions.showOperation(tf.div(c2, v2))

print('----------truediv----------')
functions.showOperation(tf.truediv(c1, v1))
functions.showOperation(tf.truediv(c2, v2))

print('----------floordiv----------')
functions.showOperation(tf.floordiv(c1, v1))
functions.showOperation(tf.floordiv(c2, v2))

print('----------mod----------')
functions.showOperation(tf.mod(c1, v1))
functions.showOperation(tf.mod(c2, v2))


############################


c1, c2, c3 = tf.constant([1,2]), tf.constant([1.0, 2.0]), tf.constant([1])
v1, v2 = tf.Variable([1,3]), tf.Variable([1.0, 3.0])

print('----------equal----------')
functions.showOperation(tf.equal(c1,v1))
functions.showOperation(tf.equal(c2,v2))

print('----------not_equal----------')
functions.showOperation(tf.not_equal(c1,v1))
functions.showOperation(tf.not_equal(c2,v2))

print('----------less----------')
functions.showOperation(tf.less(c1,v1))
functions.showOperation(tf.less(c2,v2))

print('----------less_equal----------')
functions.showOperation(tf.less_equal(c1,v1))
functions.showOperation(tf.less_equal(c2,v2))

print('----------greater----------')
functions.showOperation(tf.greater(c1,v1))
functions.showOperation(tf.greater(c2,v2))

print('----------greater_equal----------')
functions.showOperation(tf.greater_equal(c1,v1))
functions.showOperation(tf.greater_equal(c2,v2))

c4 = tf.constant([[1,3], [5,7]])
v4 = tf.Variable([[2,4], [6,8]])

cond1 = tf.Variable([[True, True], [False, False]])
cond2 = tf.Variable([[True, False], [False, True]])

print('----------where----------')
functions.showOperation(tf.where(cond1, c4, v4))
functions.showOperation(tf.where(cond2, c4, v4))

c5 = tf.constant([[True, True], [False, False]])
v5 = tf.Variable([[False, True], [True, False]])

functions.showOperation(tf.where(c5))
functions.showOperation(tf.where(v5))

functions.showOperation(tf.where(cond1, c4, v4))
functions.showOperation(tf.where(cond2, c4, v4))

print('----------and, or, xor, not----------')
functions.showOperation(tf.logical_and(c5, v5))
functions.showOperation(tf.logical_or(c5, v5))
functions.showOperation(tf.logical_xor(c5, v5))
functions.showOperation(tf.logical_not(v5))

###################################

c1 = tf.constant([1,3,5,7,9,0,2,4,6,8])
c2 = tf.constant([1,3,5])
v1 = tf.constant([[1,2,3,4,5,6],[7,8,9,0,1,2]])
v2 = tf.constant([[1,2,3], [7,8,9]])

print('----------slice----------')
functions.showOperation(tf.slice(c1, [2], [3])) # 2 to 5(= 2 + 3)
functions.showOperation(tf.slice(v1, [0,2], [1,2]))
functions.showOperation(tf.slice(v1, [0,2], [2,2]))
functions.showOperation(tf.slice(v1, [0,2], [2,-1]))





