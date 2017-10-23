# coding: utf-8

from mt_enc_dec import MTEnDecoder
from semantic_logic_model import MTSemanticLogicED
import tensorflow as tf
import corpora
from global_variable import GlobalVariable, ValidOrTestParm, config
from word2vec import WordToVector
import numpy as np
import time


def init_variables(language, isTraining=False):
    dataset = corpora.DataSet()
    if isTraining:
        dataset.loadDateSet(language, isTraining=isTraining, corpus_dir=None)

        if type(GlobalVariable.wordToVector) == type(None):
            GlobalVariable.wordToVector = WordToVector(GlobalVariable.bin_file_path)

        if type(GlobalVariable.ph_embedding) == type(None):
            GlobalVariable.ph_embedding = GlobalVariable.wordToVector.word2vector(GlobalVariable.placeholder)

        GlobalVariable.corpus_sets = dataset.train_set
        GlobalVariable.corpus_sets_num = len(GlobalVariable.corpus_sets)
        GlobalVariable.shuffle_index = np.arange(GlobalVariable.corpus_sets_num)
        np.random.shuffle(GlobalVariable.shuffle_index)

        ValidOrTestParm.corpus_sets = dataset.valid_set
        ValidOrTestParm.corpus_sets = dataset.valid_set
        ValidOrTestParm.corpus_sets_num = len(ValidOrTestParm.corpus_sets)
    else:
        dataset.loadDateSet(language, corpus_dir=None)
        ValidOrTestParm.corpus_sets = dataset.valid_set
        ValidOrTestParm.corpus_sets_num = len(ValidOrTestParm.corpus_sets)


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
      fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))
  return np.exp(costs / iters)


def main(mTSemLogModel, isTraining=True):
    x = tf.placeholder(tf.float32, [None, mTSemLogModel.n_steps, mTSemLogModel.n_inputs])

    # obtain the semantic representation code
    # _, self.semantic_rpt = self.encode(x, self.weights_ed, self.biases_ed)
    # shape of semantic_rpt:(batch_size, n_hidden_units)
    _, mTSemLogModel.semantic_rpt = mTSemLogModel.encode(x, mTSemLogModel.key_map['AE_encoder_scope'], mTSemLogModel.key_map['AE_encoder'], isTraining=isTraining)
    # obtain the logic representation code
    # shape of logic_rpt:(batch_size, n_hidden_units)
    _, mTSemLogModel.logic_rpt = mTSemLogModel.encode(x, mTSemLogModel.key_map['DL_encoder_scope'], mTSemLogModel.key_map['DL_encoder'], isTraining=isTraining)

    # concat semantic representation and logic representation as the input dialog system decoder' input tenor
    mTSemLogModel.semantic_logic_rpt = tf.concat([mTSemLogModel.semantic_rpt, mTSemLogModel.logic_rpt], 1)

    # ##########################################################################
    # This code executes single language auto-encode task.
    # ##########################################################################

    pred = mTSemLogModel.decode(mTSemLogModel.semantic_rpt, mTSemLogModel.key_map['AE_decoder_scope'], mTSemLogModel.key_map['AE_decoder'], isTraining=isTraining)

    # raw_input represent x after reshaping, shape: (batch_size, self.n_inputs * self.n_steps)
    raw_input = tf.reshape(x, (-1, mTSemLogModel.n_inputs * mTSemLogModel.n_steps))
    # shaoe of pred after reshaping: (batch_size, self.n_inputs * self.n_steps)
    pred = tf.reshape(pred, (-1, mTSemLogModel.n_inputs * mTSemLogModel.n_steps))

    # loss函数，优化方法
    cost = tf.reduce_mean(tf.pow(raw_input - pred, 2))
    train_op_optimizer = tf.train.RMSPropOptimizer(mTSemLogModel.lr).minimize(cost)

    # tf.summary.scalar("cost", cost)

    # ##########################################################################
    # This code executes single language dialog-system task.
    # ##########################################################################

    # placeholder for response sequences of shape: (batch_size, n_steps_response, embedding_len)
    y = tf.placeholder(tf.float32, [None, mTSemLogModel.n_steps_dl_out, mTSemLogModel.n_inputs])
    # pred_response_seq = self.decode(self.semantic_rpt, self.key_map['DL_decoder_scope'], self.key_map['DL_decoder'])
    # concat_times=2 ==> semantic_logic_rpt
    pred_response_seq = mTSemLogModel.decode(mTSemLogModel.semantic_logic_rpt, mTSemLogModel.key_map['DL_decoder_scope'], mTSemLogModel.key_map['DL_decoder'], concat_times=2,isTraining=isTraining)

    # raw_response_seq represent y after reshaping, stand for next sequence from corpus.
    raw_response_seq = tf.reshape(y, (-1, mTSemLogModel.n_inputs * mTSemLogModel.n_steps_dl_out))
    pred_response_seq = tf.reshape(pred_response_seq, (-1, mTSemLogModel.n_inputs * mTSemLogModel.n_steps_dl_out))

    # loss函数，优化方法
    cost_DL = tf.reduce_mean(tf.pow(raw_response_seq - pred_response_seq, 2), name='cost_dialog')
    train_op_optimizer_DL = tf.train.RMSPropOptimizer(mTSemLogModel.lr).minimize(cost_DL)

    # tf.summary.scalar("cost_dialog", cost_DL)
    saver = tf.train.Saver()

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
    GPU_config = tf.ConfigProto()
    GPU_config.gpu_options.per_process_gpu_memory_fraction = 0.99  # 程序最多只能占用指定gpu99%的显存
    GPU_config.gpu_options.allow_growth = True  # 程序按需申请内存
    with tf.Session(config=GPU_config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        if isTraining:
            step = 0
            save_count = 1
            while step * mTSemLogModel.batch_size < mTSemLogModel.training_iters:
                try:
                    batch_xs, batch_ys = corpora.corpus_next_batch(mTSemLogModel.batch_size)
                    # Training order: firstly, use common corpus to train AE, then use dialog corpus to train AE and DL
                    # And number of single AE training iteration-config['AE_completed'] is the separation of these two steps
                    if step * mTSemLogModel.batch_size <= config['AE_completed']:
                        sess.run([train_op_optimizer, cost], feed_dict={x: batch_xs})
                        if step % 200 == 0:
                            batch_val_xs, _ = corpora.get_next_batch(mTSemLogModel.batch_size, isValiding=True)
                            _, Y1_loss = sess.run([train_op_optimizer, cost], feed_dict={x: batch_val_xs})
                            print("single AE training -> single language auto-encoder loss: ", "{:.9f}".format(Y1_loss))
                        step += 1
                        continue
                    # And this is the second step, use dialog corpus to train AE and DL
                    # AE training
                    sess.run([train_op_optimizer, cost], feed_dict={ x: batch_xs })
                    if step % 200 == 0:
                        batch_val_xs, _ = corpora.get_next_batch(mTSemLogModel.batch_size, isValiding=True)
                        _, Y1_loss = sess.run([train_op_optimizer, cost], feed_dict={x: batch_val_xs})
                        print("single language auto-encoder loss: ", "{:.9f}".format(Y1_loss))
                    # DL training
                    sess.run([train_op_optimizer_DL, cost_DL], feed_dict={x: batch_xs, y: batch_ys })
                    if step % 200 == 0:
                        batch_val_xs, batch_val_ys = corpora.get_next_batch(mTSemLogModel.batch_size, isValiding=True)
                        _, Y2_loss = sess.run([train_op_optimizer_DL, cost_DL], feed_dict={x: batch_val_xs, y: batch_val_ys })
                        print("single language(English) dialog system loss: ", "{:.9f}".format(Y2_loss))

                    step += 1
                    if (step * mTSemLogModel.batch_size + 1) > (config['checkout_iters'] * save_count):
                        saver.save(sess, GlobalVariable.MTSemanticLogicED_save_path, global_step=step * mTSemLogModel.batch_size + 1)
                        save_count += 1
                except Exception as e:
                    print('Exceptions occurs when feed data in batch_size, details like this:', "Exception: {0}".format(e))
            # save_path = saver.save(sess, GlobalVariable.model_save_path)
            saver.save(sess, GlobalVariable.model_save_path, global_step=55556666)
            print("Optimization Finished!")
        else:
            # Restore model weights from previously saved model
            print("Model restored from file: %s" % GlobalVariable.model_save_path)
            saver.restore(sess, GlobalVariable.model_save_path)
            loss_count = 0
            total_loss = 0.0
            while ValidOrTestParm.epochs_completed<2:
                batch_test_xs, batch_test_ys = corpora.get_next_batch(mTSemLogModel.batch_size, isValiding=False)
                if type(batch_test_xs) != type(None):
                    loss_count += 1
                    _, DL_loss = sess.run([train_op_optimizer_DL, cost_DL], feed_dict={x: batch_test_xs, y: batch_test_ys})
                    total_loss += DL_loss
            if loss_count > 0:
                print('average loss is:', total_loss*1.0/loss_count)


if __name__ == '__main__':
    # train
    init_variables('English', isTraining=True)
    mTSemanticLogicED = MTSemanticLogicED()
    main(mTSemanticLogicED, isTraining=True)

    # test
    # init_variables('English', isTraining=False)
    # mTSemanticLogicED = MTSemanticLogicED()
    # main(mTSemanticLogicED, isTraining=False)