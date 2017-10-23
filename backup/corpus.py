# coding: utf-8
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt

class GlobalVariable(object):
    def __init__(self):
        self.vocabulary_size = 10
        self.embed_dim = 5

        # embedding layer
        self.embedding = tf.get_variable("embedding", [self.vocabulary_size, self.embed_dim], dtype=tf.float32)

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/captcha/images/', 'the directory stored image files and label file')

max_context_len = 667
max_response_len = 304
vocabulay_len = 144950


def parse_captcha_text(image_filepth):
    """
    image_filepth: 333#abc3.jpg
    """
    splits = image_filepth.split('#')
    captcha_text = splits[1].split('.')[0]
    return captcha_text


def convert2gray(img):
    if len(img.shape) > 2:
        gray_img = np.mean(img, -1)
        return gray_img
    else:
        return img
    pass


def text2vec(text):
    text_len = len(text)
    if text_len > 4:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(4 * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


def encode_to_tfrecords(train_filepth_re, data_root_dir, filename='val.tfrecords'):
    writer = tf.python_io.TFRecordWriter(data_root_dir + '/' + filename)
    num_example = 0
    with open(train_filepth_re, 'r', encoding='utf-8') as f:
        for record in f.readlines():
            splits = record.split(';')
            if len(splits) != 4:
                continue
            id = int(splits[0])
            label = int(splits[3])

            context_ids_str = splits[1].split(' ')
            context_ids_len = len(context_ids_str)
            context_ids = np.zeros(max_context_len)
            for i in range(context_ids_len):
                context_ids[i] = int(context_ids_str[i])

            response_ids_str = splits[2].split(' ')
            response_ids_len = len(response_ids_str)
            response_ids = np.zeros(max_response_len)
            for j in range(response_ids_len):
                response_ids[j] = int(response_ids_str[j])

            example = tf.train.Example(features=tf.train.Features(feature={
                'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[id])),
                'context_ids': tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_ids.tobytes()])),
                'context_ids_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[context_ids_len])),
                'response_ids': tf.train.Feature(bytes_list=tf.train.BytesList(value=[response_ids.tobytes()])),
                'response_ids_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[response_ids_len])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            serialized = example.SerializeToString()
            writer.write(serialized)
            num_example += 1
            if num_example % 1000 == 0:
                print("已有样本数据量：", num_example)
    print("最终样本数据量：", num_example)
    writer.close()


def decode_from_tfrecords(filename, num_epoch=None):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epoch, shuffle=True)
    print(filename_queue)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(serialized, features={
        'id': tf.FixedLenFeature([], tf.int64),
        'context_ids': tf.FixedLenFeature([], tf.string),
        'context_ids_len': tf.FixedLenFeature([], tf.int64),
        'response_ids': tf.FixedLenFeature([], tf.string),
        'response_ids_len': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

    context_ids = tf.decode_raw(features['context_ids'], tf.int64)
    context_ids = tf.reshape(context_ids, tf.stack([max_context_len]), name='context_ids')

    response_ids = tf.decode_raw(features['response_ids'], tf.int64)
    response_ids = tf.reshape(response_ids, tf.stack([max_response_len]), name='response_ids')

    # label = tf.decode_raw(features['label'], tf.int64)
    # label = tf.reshape(label, tf.stack(), name='label')

    return context_ids, response_ids

def get_batch(context_ids, batch_size, crop_size):
    # 生成batch
    """
    shuffle_batch函数的参数：capacity用于定义shuffle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该
                            足够大，保证数据打的足够乱
    """
    '''
    shuffle_batch()但需要注意的是它是一种图运算，要跑在sess.run()里
    '''
    context_ids_batch = tf.train.shuffle_batch([context_ids], batch_size=batch_size, num_threads=8,
                                                 capacity=20000, min_after_dequeue=5000)
    return context_ids_batch


def showImage(text, image):
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()


def test():
    train_filepth_re = '/Users/ZhangXuemiao/Desktop/dialog-data/ubuntu/val_sample.dat'
    data_root_dir = '/Users/ZhangXuemiao/Desktop/dialog-data/ubuntu'
    encode_to_tfrecords(train_filepth_re, data_root_dir, filename='val_sample.tfrecords')

def test2():
    data_root_dir = '/Users/ZhangXuemiao/Desktop/dialog-data/ubuntu/'
    # decode_from_tfrecords(data_root_dir + 'val_sample.tfrecords')
    context_ids, _ = decode_from_tfrecords(data_root_dir + 'val_sample.tfrecords')
    context_ids_batch = get_batch(context_ids, batch_size=128, crop_size=None)
    print("context_ids_batch->", context_ids_batch)

def main():
    data_root_dir = '/Users/ZhangXuemiao/Desktop/dialog-data/ubuntu/'

    context_ids, response_ids = decode_from_tfrecords(data_root_dir + 'val_sample.tfrecords')
    print('context_ids, response_ids->', context_ids, response_ids)

    context_ids_batch = get_batch(context_ids, 128, None)
    print("context_ids_batch->", context_ids_batch)
    init = tf.initialize_all_variables()

    ####################################ceshi
    globalVariable = GlobalVariable()
    globalVariable.input_ids = tf.placeholder(dtype=tf.int32, shape=[128 ,None])

    inputs = tf.nn.embedding_lookup(globalVariable.embedding, globalVariable.input_ids)
    ####################################ceshi

    with tf.Session() as session:
        # session.run(init)
        session.run(tf.local_variables_initializer())
        # session.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for l in range(4):
                '''
                每run一次，就会指向下一个样本，一直循环
                '''
                # 塞数据
                '''
                自思：
                在ssession.run之前，graph已经建好，各个node、tensor都已经定义好，但是却没有数据流通，都是静态的
                二session.run就是要使真个graph流动起来，往tensor中不断地输入数据
                '''

                '''
                image_np, label_np = session.run([image, label])
                image和label都是tensor，image_np和label_np都是tensor中的值(value)
                也就是session.run把tensor中的值取出来
                '''
                context_ids_np, response_ids_np = session.run([context_ids, response_ids ])
                print('context_ids_np.shape->',context_ids_np.shape)
                print('context_ids_np ->', context_ids_np)


                context_ids_batch_np = session.run([context_ids_batch])


                print(session.run(inputs, feed_dict={globalVariable.input_ids: context_ids_batch}))

                print(len(context_ids_batch_np[0][0]))
                print(context_ids_batch_np[0][0])
                print(context_ids_batch_np[0][0][0])
                print(int(context_ids_batch_np[0][0][0]))
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()  # queue需要关闭，否则报错
        coord.join(threads)


if __name__ == '__main__':
    main()
    # test()
    # test2()
'''
经验总结：
        要养成为每个operation增加name，如：placeholder，Varaiable，以及一些其他图操作(tf.multiply, add, shuffle_batch等)；
        为每个tensor增加name；调试的时候非常有用
'''
