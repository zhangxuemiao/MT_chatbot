# coding=utf-8
import tensorflow as tf
import numpy as np

vocabulary_size = 10
embed_dim = 5

input_ids = tf.placeholder(dtype=tf.int32, shape=[None])
#embedding layer
with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
    embedding = tf.get_variable("embedding",[vocabulary_size,embed_dim],dtype=tf.float32)
    inputs=tf.nn.embedding_lookup(embedding, input_ids)

# input_ids = tf.placeholder(dtype=tf.int32, shape=[None])
# embedding = tf.Variable(np.identity(5, dtype=np.int32))
# input_embedding = tf.nn.embedding_lookup(embedding, input_ids)
a = tf.placeholder(dtype=tf.int32, shape=[128, 200])
b = tf.placeholder(dtype=tf.int32, shape=[128, 200])
print(a)
print(b)
c = tf.placeholder(dtype=tf.int32, shape=[400])
result = tf.concat([a, b], 1)
print('result->', result)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print("embedding->", embedding.eval())
print(sess.run(inputs, feed_dict={input_ids:[1, 2, 3, 0, 3, 2, 1]}))