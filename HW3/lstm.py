#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:30:17 2017

@author: yuan
"""
import numpy as np
import tensorflow as tf
#from tensorflow.contrib import rnn
import pdb
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
imdb = np.load('imdb_word_emb.npz')
X_train = imdb['X_train']
y_train = imdb['y_train']
X_test  = imdb['X_test']
y_test  = imdb['y_test']


input_dim=128
time_step=80
n_class=2
n_hidden=128
batch_size=250
data_length=25000
learning_rate=0.01
x=tf.placeholder(tf.float32,[None,time_step,input_dim])
y=tf.placeholder(tf.float32,[None,n_class])
epoach=20

weights={'wf1': tf.Variable(tf.random_normal([n_hidden, n_class]))}
biases={'bf1': tf.Variable(tf.random_normal([n_class]))}


def LSTM(x, weights, biases):
    print(tf.shape(x))
    x = tf.unstack(x, time_step, 1)
    print(tf.shape(x))

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['wf1']) + biases['bf1']

pred = LSTM(x, weights, biases)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

init=tf.global_variables_initializer()

   # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess


#init=tf.global_variables_initializer()
session_conf=tf.ConfigProto(intra_op_parallelism_threads=8)



learning_curve=[]
with tf.Session(config=session_conf) as sess:
    sess.run(init)
    label_train =sess.run(tf.one_hot(y_train, 2))
    label_test=sess.run(tf.one_hot(y_test, 2))
    for i in range(epoach) :
        for  j in range(int(data_length/batch_size)):
            batch_x, batch_y = X_train[batch_size*j:batch_size*(j+1)],label_train[batch_size*j:batch_size*(j+1)] 
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
          #  if(j%10):
          #      acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})

             #   loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
             #   print("Iter " + str(j*batch_size) + ", Minibatch Loss= " + \
               #   "{:.6f}".format(loss) + ", Training Accuracy= " + \
                #  "{:.5f}".format(acc))
    
        acc=sess.run(accuracy,feed_dict={x:X_train , y:label_train})    
        learning_curve.append(acc)
        print("train Accuracy : ",acc)
        test_data = X_test
        test_label = label_test
        acc_test=sess.run(accuracy, feed_dict={x: test_data, y: test_label})
        print("Testing Accuracy:", acc_test)
        print("train error rate : ",1-acc_test)



learning_curve=np.ones(epoach)-learning_curve

plt.plot(range(epoach),learning_curve,label='learning_curve')
plt.show()












