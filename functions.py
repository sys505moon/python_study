#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 18:37:52 2017

@author: moon
"""

import tensorflow as tf
import functools, operator

def getLength(t):
    temp = (dim.value for dim in t.get_shape())
    return functools.reduce(operator.mul, temp)

def showConstant(t):
    sess = tf.InteractiveSession()
    print(t.eval())
    sess.close()
    
def showConstantDetail(t):
    sess = tf.InteractiveSession()
    print(t.eval())
    print('shape :', tf.shape(t))
    print('size :', tf.size(t))
    print('rank :', tf.rank(t))
    print(t.get_shape())
    
    sess.close()
    
def showVariable(v):
    sess = tf.InteractiveSession()
    v.initializer.run()
    print(v.eval())
    sess.close()
    
def var2Numpy(v):
    sess = tf.InteractiveSession()
    v.initializer.run()
    n = v.eval()
    sess.close()
    
    return n

   
def op2Numpy(op):
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    ret = sess.run(op)
    sess.close()

    return ret

def showOperation(op):
    print(op2Numpy(op))

