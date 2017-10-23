import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class Encoder(object):

    def __init__(self, input_x, config, is_training=True):
        self.keep_prob = config.keep_prob
        self.batch_size = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.hidden_neural_size = config.hidden_neural_size
        self.hidden_layer_num = config.hidden_layer_num

        num_step = config.num_step
        self.input_data = tf.placeholder(tf.int32, [None, num_step])
        self.target = tf.placeholder(tf.int64, [None])
        self.mask_x = tf.placeholder(tf.float32, [num_step, None])

        # build LSTM network
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_neural_size)
        if self.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=self.keep_prob
            )

        # ################################
        # for minist_rnn
        # ################################
        # RNN学习时使用的参数
        learning_rate = 0.001
        training_iters = 100000
        batch_size = 128
        display_step = 10

        # 神经网络的参数
        self.n_input = 28  # 输入层的n
        self.n_steps = 28  # 28长度
        self.n_hidden = 128  # 隐含层的特征数
        self.n_classes = 10  # 输出的数量，因为是分类问题，0~9个数字，这里一共有10个
        # 构建tensorflow的输入X的placeholder
        x = tf.placeholder("float", [None, self.n_steps, self.n_input])

        # tensorflow里的LSTM需要两倍于n_hidden的长度的状态，一个state和一个cell
        # Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
        istate = tf.placeholder("float", [None, 2 * self.n_hidden])
        # 输出Y
        y = tf.placeholder("float", [None, self.n_classes])
        weights = {
            'hidden': tf.Variable(tf.random_normal([self.n_input, self.n_hidden])),  # Hidden layer weights
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # MultiRNNCell
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*self.hidden_layer_num, state_is_tuple=True)
        # self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)



        self.input_x = input_x
        self.representation = None
        pass

    def encode(self, _X, _istate, _weights, _biases):
        # 规整输入的数据
        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        _X = tf.reshape(_X, [-1, self.n_input])  # (n_steps*batch_size, n_input)
        # 输入层到隐含层，第一次是直接运算
        _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
        # 之后使用LSTM
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        # 28长度的sequence，所以是需要分解位28次
        _X = tf.split(0, self.n_steps, _X)  # n_steps * (batch_size, n_hidden)
        # 开始跑RNN那部分
        outputs, states = tf.nn.rnn(lstm_cell, _X, initial_state=_istate)

        return outputs, states

        # 输出层
        # return tf.matmul(outputs[-1], _weights['out']) + _biases['out']
        pass


class Decoder(object):
    def __init__(self, representation):
        self.representation = representation
        pass

    def encode(self):

        pass