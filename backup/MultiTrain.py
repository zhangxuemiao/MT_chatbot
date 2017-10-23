# coding: utf-8

#  GRAPH CODE
# ============

# Import Tensorflow and Numpy
import tensorflow as tf
import numpy as np

# ======================
# Define the Graph
# ======================

# Define the Placeholders
X = tf.placeholder("float", [10, 10], name="X")
Y1 = tf.placeholder("float", [10, 20], name="Y1")
Y2 = tf.placeholder("float", [10, 20], name="Y2")

# Define the weights for the layers

initial_shared_layer_weights = np.random.rand(10,20)
initial_Y1_layer_weights = np.random.rand(20,20)
initial_Y2_layer_weights = np.random.rand(20,20)

shared_layer_weights = tf.Variable(initial_shared_layer_weights, name="share_W", dtype="float32")
Y1_layer_weights = tf.Variable(initial_Y1_layer_weights, name="share_Y1", dtype="float32")
Y2_layer_weights = tf.Variable(initial_Y2_layer_weights, name="share_Y2", dtype="float32")

# Construct the Layers with RELU Activations
shared_layer = tf.nn.relu(tf.matmul(X,shared_layer_weights))
Y1_layer = tf.nn.relu(tf.matmul(shared_layer,Y1_layer_weights))
Y2_layer = tf.nn.relu(tf.matmul(shared_layer,Y2_layer_weights))

# Calculate Loss
Y1_Loss = tf.nn.l2_loss(Y1-Y1_layer)
Y2_Loss = tf.nn.l2_loss(Y2-Y2_layer)

# optimisers
Y1_op = tf.train.AdamOptimizer().minimize(Y1_Loss)
Y2_op = tf.train.AdamOptimizer().minimize(Y2_Loss)

# Calculation (Session) Code
# ==========================

# open the session

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    for iters in range(10):
        if np.random.rand() < 0.5:
            _, Y1_loss = session.run([Y1_op, Y1_Loss], {
                              X: np.random.rand(10,10)*10,
                              Y1: np.random.rand(10,20)*10,
                              Y2: np.random.rand(10,20)*10
                              })
            print(Y1_loss)
        else:
            _, Y2_loss = session.run([Y2_op, Y2_Loss], {
                              X: np.random.rand(10,10)*10,
                              Y1: np.random.rand(10,20)*10,
                              Y2: np.random.rand(10,20)*10
                              })
            print(Y2_loss)