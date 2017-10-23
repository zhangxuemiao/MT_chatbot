# coding: utf-8
import tensorflow as tf
import corpora
from global_variable import config, GlobalVariable
from word2vec import WordToVector
import numpy as np

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

class MTEnDecoder(object):
    def __init__(self):

        # hyper-parameters
        self.lr = config['learning_rate']
        self.training_iters = config['max_training_iters']
        self.batch_size = config['batch_size']

        self.n_inputs = config['embedding_len']   # input (img shape: 28*28)
        self.n_steps = config['max_query_len']    # auto-encoder time steps
        self.n_steps_dl_out = config['max_response_len']  # dialog system time steps
        self.n_hidden_units = config['n_hidden_units']   # neurons in hidden layer

        # Define weights
        self.weights_ed = {
            # (28, 128)
            'encoder_in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            # (128, 10), 用于auto-encoder
            'decoder_out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_inputs*self.n_steps])),
            # 用于对话系统
            'decoder_out_dl': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_inputs * self.n_steps_dl_out]))
        }
        self.biases_ed = {
            # (128, )
            'encoder_in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            # (10, )
            'decoder_out': tf.Variable(tf.constant(0.1, shape=[self.n_inputs*self.n_steps, ])),
            # 用于对话系统
            'decoder_out_dl': tf.Variable(tf.constant(0.1, shape=[self.n_inputs * self.n_steps_dl_out, ]))
        }

        self.key_map = {
            "encoder": "encoder_in",
            "AE_decoder": "decoder_out",
            "DL_decoder": "decoder_out_dl",
            "AE_decoder_scope": "AE_decoder_scope",
            "DL_decoder_scope": "DL_decoder_scope",
        }

        self.representation = None

    def encode(self, X, weights, biases):
        # hidden layer for input to cell
        ########################################

        # transpose the inputs shape from
        # X ==> (128 batch * 28 steps, 28 inputs)
        X = tf.reshape(X, [-1, self.n_inputs])

        # into hidden
        # X_in = (128 batch * 28 steps, 128 hidden)
        X_in = tf.matmul(X, weights['encoder_in']) + biases['encoder_in']
        # print('1-->', X_in)
        # X_in ==> (128 batch, 28 steps, 128 hidden)
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])
        # print('2-->', X_in)

        # basic LSTM Cell.
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        else:
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)
        # lstm cell is divided into two parts (c_state, h_state)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # You have 2 options for following step.
        # 1: tf.nn.rnn(cell, inputs);
        # 2: tf.nn.dynamic_rnn(cell, inputs).
        # If use option 1, you have to modified the shape of X_in, go and check out this:

        # In here, we go for option 2.
        # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
        # Make sure the time_major is changed accordingly.
        with tf.variable_scope('encoder_rnn'):
            outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

        return final_state

    def autoencoder_decode(self, representation, weights, biases):
        # raw representation =>(128 batch_size, 128 hidden_units)
        # reshape & transpose representation shape from (128 batch_size, 1 steps, 28 hidden_units)
        representation = tf.reshape(representation, [-1, self.batch_size, self.n_hidden_units])
        representation = tf.transpose(representation, [1, 0, 2])

        # basic LSTM Cell.
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        else:
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)
        # lstm cell is divided into two parts (c_state, h_state)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # You have 2 options for following step.
        # 1: tf.nn.rnn(cell, inputs);
        # 2: tf.nn.dynamic_rnn(cell, inputs).
        # If use option 1, you have to modified the shape of X_in, go and check out this:
        # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
        # In here, we go for option 2.
        # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
        # Make sure the time_major is changed accordingly.
        with tf.variable_scope('decoder_rnn'):
            outputs, final_state = tf.nn.dynamic_rnn(cell, representation, initial_state=init_state, time_major=False)
        # final_state[1] == outputs[-1]

        # hidden layer for output as the final results
        #############################################
        # results = tf.matmul(final_state[1], weights['out']) + biases['out']

        # # or
        # unpack to list [(batch, outputs)..] * steps
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
        else:
            outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        results = tf.matmul(outputs[-1], weights['decoder_out']) + biases['decoder_out']  # shape = (128, 10)
        return results


    def decode(self, representation, name_scope, weights_biases_key):
        """
        Decode
        :param representation:
        :param weights_biases_key: if this function use for auto-encoder, then weights_biases_key should be "decoder_out",
                                    if use for dialog system, then weights_biases_key should be "decoder_out_dl"
        :return:
        """

        representation = tf.reshape(representation, [-1, self.batch_size, self.n_hidden_units])
        representation = tf.transpose(representation, [1, 0, 2])

        # basic LSTM Cell.
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        else:
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)
        # lstm cell is divided into two parts (c_state, h_state)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        with tf.variable_scope(name_scope):
            outputs, final_state = tf.nn.dynamic_rnn(cell, representation, initial_state=init_state, time_major=False)

        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
        else:
            outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

        # print('outputs->', outputs)
        # print('outputs[-1]->', outputs[-1])
        # print('final_state->', final_state)
        results = tf.matmul(outputs[-1], self.weights_ed[weights_biases_key]) + self.biases_ed[weights_biases_key]  # shape = (128, 10)
        return results


    def train(self):
        x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])

        # obtain the semantic representation code
        _, self.representation = self.encode(x, self.weights_ed, self.biases_ed)

        # print("semantic representation ->", self.representation)

        # ##########################################################################
        # This code executes single language auto-encode task.
        # ##########################################################################

        pred = self.decode(self.representation, self.key_map['AE_decoder_scope'], self.key_map['AE_decoder'])

        # raw_input represent x after reshaping
        raw_input = tf.reshape(x, (-1, self.n_inputs*self.n_steps))
        pred = tf.reshape(pred, (-1, self.n_inputs*self.n_steps))

        # print('pred->', pred)
        # print('raw_input->', raw_input)

        # loss函数，优化方法
        cost = tf.reduce_mean(tf.pow(raw_input - pred, 2))
        train_op_optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(cost)

        # ##########################################################################
        # This code executes single language dialog-system task.
        # ##########################################################################

        # placeholder for response sequences of shape: (batch_size, n_steps_response, embedding_len)
        y = tf.placeholder(tf.float32, [None, self.n_steps_dl_out, self.n_inputs])
        pred_response_seq = self.decode(self.representation, self.key_map['DL_decoder_scope'], self.key_map['DL_decoder'])

        # raw_response_seq represent y after reshaping, stand for next sequence from corpus.
        raw_response_seq = tf.reshape(y, (-1, self.n_inputs * self.n_steps_dl_out))
        pred_response_seq = tf.reshape(pred_response_seq, (-1, self.n_inputs * self.n_steps_dl_out))

        # print('y->', y)
        # print('pred_response_seq->', pred_response_seq)
        # print('raw_response_seq->', raw_response_seq)

        # loss函数，优化方法
        cost_DL = tf.reduce_mean(tf.pow(raw_response_seq - pred_response_seq, 2))
        train_op_optimizer_DL = tf.train.RMSPropOptimizer(self.lr).minimize(cost_DL)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # tf.initialize_all_variables() no long valid from
            # 2017-10-16 if using tensorflow >= 0.12
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            while step * self.batch_size < self.training_iters:
                try:
                    batch_xs, batch_ys = corpora.corpus_next_batch(self.batch_size)

                    # without random
                    _, Y1_loss = sess.run([train_op_optimizer, cost], feed_dict={ x: batch_xs })
                    if step % 20 == 0: print("single language auto-encoder loss: ", "{:.9f}".format(Y1_loss))
                    _, Y2_loss = sess.run([train_op_optimizer_DL, cost_DL], feed_dict={
                        x: batch_xs,
                        y: batch_ys
                    })
                    if step % 20 == 0: print("single language(English) dialog system loss: ", "{:.9f}".format(Y2_loss))

                    # with random
                    # if np.random.rand() < 0.5:
                    #     _, Y1_loss = sess.run([train_op_optimizer, cost], feed_dict={
                    #         x: batch_xs
                    #     })
                    #     if step % 20 == 0:
                    #         print("single language auto-encoder loss: ", "{:.9f}".format(Y1_loss))
                    # else:
                    #     _, Y2_loss = sess.run([train_op_optimizer_DL, cost_DL], feed_dict={
                    #         x: batch_xs,
                    #         y: batch_ys
                    #     })
                    #     if step % 20 == 0:
                    #         print("single language(English) dialog system loss: ", "{:.9f}".format(Y2_loss))
                    step += 1
                except Exception as e:
                    print('Exceptions occurs when feed data in batch_size, details like this:', "Exception: {0}".format(e))
            # save_path = saver.save(sess, GlobalVariable.model_save_path)
            saver.save(sess, GlobalVariable.model_save_path)
        print("Optimization Finished!")