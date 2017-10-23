# -*- coding: utf-8 -*-

import tensorflow as tf
# from tensorflow.contrib.rnn.python.ops import rnn_cell
# from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn


# lstm_cell = rnn.BasicLSTMCell(rnn_size)
# outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

import numpy as np

x=tf.placeholder(dtype=tf.float64,shape=[10,10,10],name="x")

train_x = np.ones(shape=[10, 10, 10], dtype=float)


cell=tf.contrib.rnn.BasicLSTMCell(10)

unstack_x = tf.unstack(x, 10, 1)

lstm_cell = rnn.BasicLSTMCell(10, forget_bias=1.0)

outputs, states = rnn.static_rnn(lstm_cell, unstack_x, dtype=tf.float64)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    outputs_array=(sess.run(outputs,feed_dict={x:train_x}))