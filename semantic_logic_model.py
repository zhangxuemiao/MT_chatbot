# coding: utf-8
import tensorflow as tf
import corpora
from global_variable import config, GlobalVariable

# set random seed for comparing the two result calculations
tf.set_random_seed(1)
# tf.reset_default_graph()

class MTSemanticLogicED(object):
    def __init__(self):

        # hyper-parameters
        self.lr = config['learning_rate']
        self.training_iters = config['max_training_iters']
        self.batch_size = config['batch_size']
        self.keep_prob = config['keep_prob']

        self.n_inputs = config['embedding_len']  # input (shape: embedding_len)
        self.n_steps = config['max_query_len']  # auto-encoder time steps
        self.n_steps_dl_out = config['max_response_len']  # dialog system time steps
        self.n_hidden_units = config['n_hidden_units']  # neurons in hidden layer
        self.n_hidden_layers = config['n_hidden_layers']

        # Define weights
        self.weights_ed = {
            # 用于自编码器 (28, 128)
            'encoder_in_ae': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            # 用于对话系统
            'encoder_in_dl': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            # (128, 10), 用于auto-encoder
            'decoder_out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_inputs * self.n_steps])),
            # 用于对话系统
            'decoder_out_dl': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_inputs * self.n_steps_dl_out]))
        }
        self.biases_ed = {
            # (128, )
            'encoder_in_ae': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            # 用于对话系统
            'encoder_in_dl': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            # (10, )
            'decoder_out': tf.Variable(tf.constant(0.1, shape=[self.n_inputs * self.n_steps, ])),
            # 用于对话系统
            'decoder_out_dl': tf.Variable(tf.constant(0.1, shape=[self.n_inputs * self.n_steps_dl_out, ]))
        }

        self.key_map = {
            "AE_encoder": "encoder_in_ae",

            "DL_encoder": "encoder_in_dl",

            "AE_decoder": "decoder_out",
            "DL_decoder": "decoder_out_dl",

            "AE_encoder_scope": "AE_encoder_scope",
            "DL_encoder_scope": "DL_encoder_scope",

            "AE_decoder_scope": "AE_decoder_scope",
            "DL_decoder_scope": "DL_decoder_scope",
        }

        # semantic representation
        self.semantic_rpt = None
        # logic representation
        self.logic_rpt = None
        # concat semantic representation and logic representation to semantic_logic_rpt
        self.semantic_logic_rpt = None


    def lstm_cell(self, isTraining):
        # lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        # drop = tf.contrib.rnn.DropoutWrapper(lstm)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units, forget_bias=0.0, state_is_tuple=True)
        if isTraining and self.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=self.keep_prob
            )
        return lstm_cell
    # def encode(self, X, weights, biases, encoder_name_scope, weights_biases_key):
    def encode(self, X, encoder_name_scope, weights_biases_key, isTraining=True):
        """

        :param X: shape of raw input tensor X: (batch_size, n_steps, embedding_len)
        :param encoder_name_scope:
        :param weights_biases_key:
        :return:
        """
        # hidden layer for input to cell
        ########################################

        # transpose the inputs shape from
        # X ==> (batch_size * max_query_len steps, embedding inputs)
        X = tf.reshape(X, [-1, self.n_inputs])

        # into hidden
        # X_in = (128 batch * max_query_len steps, 128 hidden)
        # X_in = tf.matmul(X, weights['encoder_in']) + biases['encoder_in']
        X_in = tf.matmul(X, self.weights_ed[weights_biases_key]) + self.biases_ed[weights_biases_key]
        # X_in ==> (128 batch_size, max_query_len steps, 128 n_hidden_units)
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])

        cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell(isTraining) for _ in range(self.n_hidden_layers)], state_is_tuple=True)
        # lstm cell is divided into two parts (c_state, h_state)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        with tf.variable_scope(encoder_name_scope):
            # shape of state: (n_hidden_layers, batch_size, n_hidden_units)
            outputs, state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False, scope='lstm_cell')
            # h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]
            # state[-1],n_hidden_layers个 LSTMStateTuple，取最后一个，LSTMStateTuple->LSTMStateTuple(c=<tf.Tensor 'AE_encoder_scope/lstm_cell/while/Exit_10:0' shape=(128, 128) dtype=float32>, h=<tf.Tensor 'AE_encoder_scope/lstm_cell/while/Exit_11:0' shape=(128, 128) dtype=float32>)
            h_state = state[-1]

        print('h_state->', h_state)
        return h_state

    def decode(self, representation, name_scope, weights_biases_key, concat_times=1, isTraining=True):
        """
        Decode
        :param representation: shape of semantic_rpt:(batch_size, n_hidden_units);
                               shape of semantic_logic_rpt_concat:(batch_size, 2*n_hidden_units)
        :param weights_biases_key: if this function use for auto-encoder, then weights_biases_key should be "decoder_out",
                                    if use for dialog system, then weights_biases_key should be "decoder_out_dl"
        :return:
        """

        print('representation->', representation)


        # if concat_times==1 ==> stand for that representation is semantic_rpt
        # else if concat_times==2 ==> stand for that representation is semantic_logic_rpt_concat
        representation = tf.reshape(representation, [-1, self.batch_size, self.n_hidden_units*concat_times])

        print('representation 1->', representation)

        representation = tf.transpose(representation, [1, 0, 2])

        print('representation 2->', representation)

        # basic LSTM Cell.
        cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell(isTraining) for _ in range(self.n_hidden_layers)], state_is_tuple=True)
        # lstm cell is divided into two parts (c_state, h_state)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        with tf.variable_scope(name_scope):
            outputs, final_state = tf.nn.dynamic_rnn(cell, representation, initial_state=init_state, time_major=False)
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        results = tf.matmul(outputs[-1], self.weights_ed[weights_biases_key]) + self.biases_ed[weights_biases_key]  # shape = (128, 10)
        return results