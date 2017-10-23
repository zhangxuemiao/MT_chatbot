# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
This code is a modified version of the code from this link:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
His code is a very good one for RNN beginners. Feel free to check it out.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

class MinistEnDecoder(object):
    def __init__(self):
        # hyperparameters
        self.lr = 0.001
        self.training_iters = 100000
        self.batch_size = 128

        self.n_inputs = 28   # MNIST data input (img shape: 28*28)
        self.n_steps = 28    # time steps
        self.n_hidden_units = 128   # neurons in hidden layer
        self.n_classes = 10      # MNIST classes (0-9 digits)

        # Define weights
        self.weights = {
            # (28, 128)
            'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            # (128, 10)
            'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        }
        self.biases = {
            # (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            # (10, )
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
        }

        self.weights_d = {
            # (28, 128)
            'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            # (128, 10)
            'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        }
        self.biases_d = {
            # (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            # (10, )
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
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
        X_in = tf.matmul(X, weights['in']) + biases['in']
        print('1-->', X_in)
        # X_in ==> (128 batch, 28 steps, 128 hidden)
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])
        print('2-->', X_in)

        # cell
        ##########################################

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
        with tf.variable_scope('encoder_rnn'):
            outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
        # zxm-> final_state[1] = outputs[-1]
        # zxm-> outputs type=> (28 steps,128 batch, 128 hidden)

        # hidden layer for output as the final results
        #############################################
        # results = tf.matmul(final_state[1], weights['out']) + biases['out']

        # # or
        # unpack to list [(batch, outputs)..] * steps
        # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        #     outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
        # else:
        #     outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        # print(outputs)
        # print(outputs[-1])
        # results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)

        return final_state

    def decode(self, representation, weights, biases):
        # hidden layer for input to cell
        ########################################

        # raw representation =>(28 steps, 128 batch_size, 128 hidden_units)
        # transpose representation shape from (128 batch_size, 1 steps, 28 hidden_units)
        representation = tf.reshape(representation, [-1, self.batch_size, self.n_hidden_units])
        representation = tf.transpose(representation, [1, 0, 2])

        # cell
        ##########################################

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
        # final_state[1] = outputs[-1]

        # hidden layer for output as the final results
        #############################################
        # results = tf.matmul(final_state[1], weights['out']) + biases['out']

        # # or
        # unpack to list [(batch, outputs)..] * steps
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
        else:
            outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)

        return results

    def train(self):
        x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        y = tf.placeholder(tf.float32, [None, self.n_classes])
        _, representation = self.encode(x, self.weights, self.biases)
        self.representation = representation
        pred = self.decode(self.representation, self.weights_d, self.biases_d)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(cost)

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            # tf.initialize_all_variables() no long valid from
            # 2017-03-02 if using tensorflow >= 0.12
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            while step * self.batch_size < self.training_iters:
                batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
                batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
                sess.run([train_op], feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                })
                if step % 20 == 0:
                    print(sess.run(accuracy, feed_dict={
                        x: batch_xs,
                        y: batch_ys,
                    }))
                step += 1


def main():
    ministEnDecoder = MinistEnDecoder()
    ministEnDecoder.train()

if __name__ == '__main__':
    main()