# coding:utf-8

# from compiler.ast import flatten
import os

import numpy as np
from flatten import flatten

from backup.category_dict import get_category_map
from global_variable import config, GlobalVariable

category_dict = get_category_map()


def get_batch(all_lines, batch_size, count, label):
    batch_xs = []
    batch_ys = []

    all_lines_len = all_lines.__len__()
    rest_lines_len = all_lines_len - count

    temp = int((rest_lines_len*1.0)/batch_size)
    if temp > 1:
        for ind in range(count,count+batch_size):
            batch_xs.append(all_lines[ind])
            batch_ys.append(label)
        count += batch_size
    else:
        for ind in range(count,all_lines_len):
            batch_xs.append(all_lines[ind])
            batch_ys.append(label)
        count = all_lines_len

    return batch_xs, batch_ys, count


def corpus_next_batch(batch_size):
    """Return the next `batch_size` examples from this data set."""

    start = GlobalVariable.index_in_epoch
    GlobalVariable.index_in_epoch += batch_size
    if GlobalVariable.index_in_epoch >= GlobalVariable.corpus_sets_num:  # epoch中的句子下标是否大于所有语料的个数，如果为True,开始新一轮的遍历

        # 回显处于第几次epoch
        print("epochs_completed->" + str(GlobalVariable.epochs_completed))

        # Finished epoch
        GlobalVariable.epochs_completed += 1
        GlobalVariable.shuffle_index = None

        # Shuffle the data
        GlobalVariable.shuffle_index = np.arange(GlobalVariable.corpus_sets_num)  # arange函数用于创建等差数组
        np.random.shuffle(GlobalVariable.shuffle_index)  # 打乱

        # Start next epoch
        start = 0
        GlobalVariable.index_in_epoch = batch_size
        # assert batch_size <= GlobalVariable.corpus_sets_num
    end = GlobalVariable.index_in_epoch
    result = [GlobalVariable.corpus_sets[GlobalVariable.shuffle_index[ind]] for ind in range(start, end)]

    batch_xs = []
    batch_ys = []
    i = 0
    for record in result:
        try:
            batch_xs.append(record[20:])
            batch_ys.append(record[:20])
            # xs_temp = [float(a) for a in record[20:]]
            # ys_temp = [int(b) for b in record[:20]]
            # batch_xs.append(xs_temp)
            # batch_ys.append(ys_temp)
            # xs_temp = None
            # ys_temp =None
        except Exception as e:
            pass

    return batch_xs, batch_ys


def get_test_corpus():
    test_record_num = GlobalVariable.test_corpus.__len__()
    result = [GlobalVariable.test_corpus[ind] for ind in range(test_record_num)]
    batch_test_xs = []
    batch_test_ys = []
    i = 0
    for record in result:
        try:
            batch_test_xs.append(map(np.float32, record[20:]))
            batch_test_ys.append(map(np.int, record[:20]))
        except Exception :
            i += 1
            print("Exception->"+str(i))

    return batch_test_xs, batch_test_ys


def next_batch(file_path, batch_size, label, vector_num=config['n_input']):
    line_content = []
    batch_xs = []
    batch_ys = []

    with open(file_path) as f:
        for line in f:
            word_vec = map(np.float32, line.strip().split(','))
            # word_vec = line.strip().split(',')
            line_content.append(word_vec)

    total_words_num = line_content.__len__()
    total = batch_size * vector_num
    # ruguo zong dancishu bi yipi de zong xiangliangshu yaoxiao,ze neng tiaochong duoshao suan duoshao
    if total > total_words_num:
        # for index in range(total_words_num,total):
        batch_size = (total_words_num*1.0) / vector_num
        if batch_size > 1:
            batch_size = int(batch_size)
        else:
            batch_size = 1
    #            jiang bugou de bufen tianchong wanzheng,ru jiang yuanlai de 150 ge danci tianchong cheng 200 ge danci
            i = 0
            for index in range(total_words_num,vector_num):
                line_content.append(line_content[vector_num-i])
                i = i + 1

    total = batch_size * vector_num

    temp_all_vector = []
    for i in range(total):
        if (i+1) % vector_num == 0:
            # print i
            if i+1 == vector_num:
                temp_all_vector.append(line_content[i])
            batch_xs.append(flatten(temp_all_vector))
            temp_all_vector = []
        temp_all_vector.append(line_content[i])

        label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    for index in range(batch_size):
        batch_ys.append(label)
    return batch_xs, batch_ys


def get_csv_files(root_dir):
    file_list = []
    for rt, dirs, files in os.walk(root_dir):
        file_list.append(files)
    return flatten(file_list)


def load_whole_file():
    all_line = []


def main():
    root_dir = '/home/zhangxuemiao/Desktop/allCategories/'
    file_list = get_csv_files(root_dir)

    print(file_list.__len__())

    for file_name in file_list:
        file_path = root_dir+file_name
        shotname, _ = os.path.splitext(file_name)
        label = category_dict[shotname]
        print(label)

    # a_t, b_t = next_batch('/home/zhangxuemiao/Desktop/alt.atheism/51060b', 30)
    # print(a_t[4].__len__(), b_t[4].__len__())
    # # print a_t


if __name__ == "__main__":
    batch_xs, batch_ys = corpus_next_batch(20)
    for i in range(batch_xs.__len__()):
        print(batch_xs[i], batch_ys[i])