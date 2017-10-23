# -*- coding: utf-8 -*-


# 利用ask/answer的训练集输入神经网络，并使用ask/answer测试向量映射集实现BP反馈与，
# 使用一个三层神经网络，让tensorflow自动调整权重参数，获得一个ask-?的模型

import tensorflow as tf  # 0.12
from tensorflow.models.rnn.translate import seq2seq_model
import os
import numpy as np
import math

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# ask/answer conversation vector file
train_ask_vec_file = 'train_ask.vec'
train_answer_vec_file = 'train_answer.vec'
test_ask_vec_file = 'test_ask.vec'
test_answer_vec_file = 'test_answer.vec'

# word table 6000
vocabulary_ask_size = 6000
vocabulary_answer_size = 6000

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
layer_size = 256
num_layers = 3
batch_size = 64


# read *dencode.vec和*decode.vec data into memory
def read_data(source_path, target_path, max_size=None):
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

if __name__ == '__main__':
    model = seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_ask_size,
                                       target_vocab_size=vocabulary_answer_size,
                                       buckets=buckets, size=layer_size, num_layers=num_layers, max_gradient_norm=5.0,
                                       batch_size=batch_size, learning_rate=0.5, learning_rate_decay_factor=0.97,
                                       forward_only=False)

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'  # forbidden out of memory

    with tf.Session(config=config) as sess:
        # 恢复前一次训练
        ckpt = tf.train.get_checkpoint_state('.')
        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        train_set = read_data(train_ask_vec_file, train_answer_vec_file)
        test_set = read_data(test_ask_vec_file, test_answer_vec_file)

        train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]

        loss = 0.0
        total_step = 0
        previous_losses = []
        # continue train，save modle after a decade of time
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

            loss += step_loss / 500
            total_step += 1

            print(total_step)
            if total_step % 500 == 0:
                print(model.global_step.eval(), model.learning_rate.eval(), loss)

                # if model has't not improve，decrese the learning rate
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # save model
                checkpoint_path = "chatbot_seq2seq.ckpt"
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                loss = 0.0
                # evaluation the model by test dataset
                for bucket_id in range(len(buckets)):
                    if len(test_set[bucket_id]) == 0:
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print(bucket_id, eval_ppx)