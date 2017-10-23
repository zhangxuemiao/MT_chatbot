# -*- coding: utf-8 -*-

import tensorflow as tf  # 0.12
from tensorflow.models.rnn.translate import seq2seq_model
import os
import sys
import locale
import numpy as np

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

train_ask_vocabulary_file = "train_ask_vocabulary.vec"
train_answer_vocabulary_file = "train_answer_vocabulary.vec"


def read_vocabulary(input_file):
    tmp_vocab = []
    with open(input_file, "r") as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    return vocab, tmp_vocab


if __name__ == '__main__':
    vocab_en, _, = read_vocabulary(train_ask_vocabulary_file)
    _, vocab_de, = read_vocabulary(train_answer_vocabulary_file)

    # word table 6000
    vocabulary_ask_size = 6000
    vocabulary_answer_size = 6000

    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    layer_size = 256
    num_layers = 3
    batch_size = 1

    model = seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_ask_size,
                                       target_vocab_size=vocabulary_answer_size,
                                       buckets=buckets, size=layer_size, num_layers=num_layers, max_gradient_norm=5.0,
                                       batch_size=batch_size, learning_rate=0.5, learning_rate_decay_factor=0.99,
                                       forward_only=True)
    model.batch_size = 1

    with tf.Session() as sess:
        # restore last train
        ckpt = tf.train.get_checkpoint_state('.')
        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("model not found")

        while True:
            input_string = raw_input('me > ').decode(sys.stdin.encoding or locale.getpreferredencoding(True)).strip()
            # 退出
            if input_string == 'quit':
                exit()

            # convert the user's input to vector
            input_string_vec = []
            for words in input_string.strip():
                input_string_vec.append(vocab_en.get(words, UNK_ID))
            bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(input_string_vec)])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(input_string_vec, [])]},
                                                                             bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            if EOS_ID in outputs:
                outputs = outputs[:outputs.index(EOS_ID)]

            response = "".join([tf.compat.as_str(vocab_de[output]) for output in outputs])
            print('AI > ' + response)