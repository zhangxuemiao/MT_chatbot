# coding: utf-8

import glob
import json
import os
import nltk
import numpy as np
from nltk.corpus import stopwords

from global_variable import config, GlobalVariable, ValidOrTestParm
# import main_operation


file_dir = GlobalVariable.ubuntu_dialogs_path
standard_file_name = "*.tsv"

class Sentence2Words(object):
    # 分割成句子
    def sen_token(self, raw):
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_tokenizer.tokenize(raw)
        return sents

    # 句子内容的清理，去掉数字标点和非字母字符
    def clean_lines(self, line):
        cleanLine = ''.join([x for x in line if x.isalpha() or x == ' '])
        return cleanLine

    # nltk.word_tokenize分词
    def word_tokenize(self, sent):  # 将单句字符串分割成词
        wordsInStr = nltk.word_tokenize(sent)
        return wordsInStr

    def clean_words(self, wordsInStr):  # 去掉标点符号，长度小于3的词以及non-alpha词，小写化
        cleanWords = []
        stopwords_en = stopwords.words('english')
        for words in wordsInStr:
            cleanWords += [[w.lower() for w in words if w.lower() not in stopwords_en and 1 <= len(w)]]
        return cleanWords

    def standard_operation(self, sentence):
        sentence = self.clean_lines(sentence)
        words = self.word_tokenize(sentence)
        # words = self.clean_words(words)
        return words


class DataSet(object):
    def __init__(self):
        self.dataSetPath = None

        self.train_rate = None
        self.test_rate = None

        self.train_set = []
        self.test_set = []
        self.valid_set = []

    # split dataset from the whole dataset into thres part: train, valid, and test dataset
    def splitDataSet(self, language):
        with open(self.dataSetPath, 'r') as json_records:
            sentence2Words = Sentence2Words()
            corpus = []
            for line in json_records:
                corpus.append(line)
            print(len(corpus))
            corpus_len = len(corpus)
            train_set_len = int(corpus_len*self.train_rate)
            test_set_len = int(corpus_len*self.test_rate)

            self.train_set = corpus[:train_set_len]
            self.test_set = corpus[train_set_len: train_set_len+test_set_len]
            self.valid_set = corpus[train_set_len+test_set_len: ]

            dataset_save_dir = './corpus/'+language
            if not os.path.exists(dataset_save_dir):
                os.mkdir(dataset_save_dir)

            with open(dataset_save_dir + '/'+language+'_train.json', 'w') as train_json_records:
                for line in self.train_set:
                    train_json_records.write(line)
            with open(dataset_save_dir + '/'+language+'_valid.json', 'w') as valid_json_records:
                for line in self.valid_set:
                    valid_json_records.write(line)
            with open(dataset_save_dir + '/'+language+'_test.json', 'w') as test_json_records:
                for line in self.test_set:
                    test_json_records.write(line)


    def loadDateSet(self, language, isTraining=False, corpus_dir=None):
        if corpus_dir == None:
            corpus_dir = './corpus/'+language+'/'
        else:
            if not os.path.exists(corpus_dir):
                assert "Directory is not exist!"
        if corpus_dir[-1] !='/':
            corpus_dir = corpus_dir + '/'
        sentence2Words = Sentence2Words()
        if isTraining:
            train_set_path = corpus_dir + language+'_train.json'
            valid_set_path = corpus_dir + language + '_valid.json'

            with open(train_set_path, 'r') as train_set_file:
                for line in train_set_file.readlines():
                    data = json.loads(line)
                    singleQuery = sentence2Words.word_tokenize(' '.join(data['q']))
                    singleResponse = sentence2Words.word_tokenize(' '.join(data['r']))
                    singleRecord = {
                        'q': singleQuery,
                        'r': singleResponse
                    }
                    self.train_set.append(singleRecord)

            with open(valid_set_path, 'r') as valid_set_file:
                for line in valid_set_file.readlines():
                    data = json.loads(line)
                    singleQuery = sentence2Words.word_tokenize(' '.join(data['q']))
                    singleResponse = sentence2Words.word_tokenize(' '.join(data['r']))
                    singleRecord = {
                        'q': singleQuery,
                        'r': singleResponse
                    }
                    self.valid_set.append(singleRecord)

        if not isTraining:
            test_set_path = corpus_dir + language + '_test.json'

            with open(test_set_path, 'r') as test_set_file:
                for line in test_set_file.readlines():
                    data = json.loads(line)
                    singleQuery = sentence2Words.word_tokenize(' '.join(data['q']))
                    singleResponse = sentence2Words.word_tokenize(' '.join(data['r']))
                    singleRecord = {
                        'q': singleQuery,
                        'r': singleResponse
                    }
                    self.test_set.append(singleRecord)


def QR_records_gen(corpus_file_path):
    sentence2Words = Sentence2Words()
    with open(corpus_file_path, 'w') as json_records:
        record_count = 0

        exit_flag = False
        fs = glob.iglob(file_dir + standard_file_name)
        for file_path in fs:
            # print(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                singlePairFlag = 0
                quizzer = None
                singleQuery = []
                singleResponse = []

                currentPerson = None
                previousPerson = None
                QR_record = { "q": None, 'r': None }

                for line in f.readlines():
                    splits = line.split("	")
                    if len(splits) != 4:
                        print(file_path, splits)
                        continue

                    currentPerson = splits[1]
                    if splits[2] == '':
                        quizzer = currentPerson

                    if currentPerson != previousPerson:
                        singlePairFlag += 1 # 1 2 3 - 1 2 3 - 1
                        if singlePairFlag >= 3:
                            singlePairFlag = 1
                            QR_record['q'] = singleQuery
                            QR_record['r'] = singleResponse

                            json_records.write(json.dumps(QR_record) + '\n')
                            # print(QR_record)
                            record_count += 1
                            if record_count % 3000 == 0:
                                print('Has store ', record_count, 'query-response records')

                            # generate mini corpus when needing
                            if record_count >= 9000:
                                exit_flag = True
                                break

                            singleQuery = []
                            singleResponse = []
                        previousPerson = currentPerson

                    if currentPerson == quizzer:
                        words = sentence2Words.standard_operation(splits[3])
                        if len(words) <= 0:
                            words.append('placeholder')
                        singleQuery.append(' '.join(words))

                    if currentPerson != quizzer:
                        words = sentence2Words.standard_operation(splits[3])
                        if len(words) <= 0:
                            words.append('placeholder')
                        singleResponse.append(' '.join(words))

            # Collaboration with inner-level break statements
            if exit_flag:
                break


def load_entire_corpus(corpus_file_path):
    with open(corpus_file_path, 'r') as json_records:
        sentence2Words = Sentence2Words()
        corpus = []
        for line in json_records:
            data = json.loads(line)
            singleQuery = sentence2Words.word_tokenize(' '.join(data['q']))
            # singleQueryWords = []
            # for seq in data['q']:
            #     wor
            singleResponse = sentence2Words.word_tokenize(' '.join(data['r']))
            singleRecord = {
                'q': singleQuery,
                'r': singleResponse
            }
            corpus.append(singleRecord)
        # print(len(corpus))
        return corpus


def sentence_clean(sentence_):
    sentence = """hi, i installed xubuntu on my laptop. i cant use wifi because it says "wireless disabled by hardware switch" -  i tried rfkill ublock as suggested in many forums, but i does not help. i have a w-lan button, but it is not recognized by windows. what can i do?\n"""
    if len(sentence) <= 0:
        return None
    sentence2Words = Sentence2Words()
    print(sentence2Words.standard_operation(sentence))


def corpus_next_batch(batch_size):
    """Return the next `batch_size` examples from this data set."""

    start = GlobalVariable.index_in_epoch
    GlobalVariable.index_in_epoch += batch_size
    if GlobalVariable.index_in_epoch >= GlobalVariable.corpus_sets_num:  # epoch中的句子下标是否大于所有语料的个数，如果为True,开始新一轮的遍历

        # 回显处于第几次epoch
        print("corpus has been trained completely->", GlobalVariable.epochs_completed, 'epochs')

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

    qr_records = [GlobalVariable.corpus_sets[GlobalVariable.shuffle_index[ind]] for ind in range(start, end)]

    batch_xs = []
    batch_ys = []
    for record in qr_records:
        try:
            q_word_embedding = []
            # print(len(record['q']))
            q_len = len(record['q'])
            if q_len < config['max_query_len']:
                for i in range(q_len, config['max_query_len']):
                    record['q'].append(GlobalVariable.placeholder)

            if q_len > config['max_query_len']:
                record['q'] = record['q'][0: config['max_query_len']-1]

            for q_word in record['q']:
                if q_word == GlobalVariable.placeholder:
                    word_embedding = GlobalVariable.ph_embedding
                else:
                    word_embedding = GlobalVariable.wordToVector.word2vector(q_word)
                if type(word_embedding) == type(None):
                    word_embedding = GlobalVariable.ph_embedding
                q_word_embedding.append(word_embedding)

            r_word_embedding = []
            r_len = len(record['r'])
            if r_len < config['max_response_len']:
                for i in range(r_len, config['max_response_len']):
                    record['r'].append(GlobalVariable.placeholder)

            if r_len > config['max_response_len']:
                record['q'] = record['q'][0: config['max_response_len'] - 1]

            # print(record['r'])

            for r_word in record['r']:
                if r_word == GlobalVariable.placeholder:
                    word_embedding = GlobalVariable.ph_embedding
                else:
                    word_embedding = GlobalVariable.wordToVector.word2vector(r_word)
                if type(word_embedding) == type(None):
                    word_embedding = GlobalVariable.ph_embedding
                r_word_embedding.append(word_embedding)

            # print(len(record['q']), len(record['r']))

            batch_xs.append(q_word_embedding)
            batch_ys.append(r_word_embedding)

        except Exception as e:
            print('corpus Error, when execute function corpus_next_batch.', "Exception: {0}".format(e))
    return batch_xs, batch_ys


def get_next_batch(batch_size, isValiding=False):
    """Return the next `batch_size` examples from this data set."""

    start = ValidOrTestParm.index_in_epoch
    ValidOrTestParm.index_in_epoch += batch_size
    # epoch中的句子下标是否大于所有语料的个数，如果为True,开始新一轮的遍历
    if ValidOrTestParm.index_in_epoch >= ValidOrTestParm.corpus_sets_num:
        # Finished one epoch
        ValidOrTestParm.epochs_completed += 1
        if not isValiding:
            ValidOrTestParm.index_in_epoch = ValidOrTestParm.corpus_sets_num
        else:
            # Start next epoch
            start = 0
            ValidOrTestParm.index_in_epoch = batch_size
    end = ValidOrTestParm.index_in_epoch

    qr_records = [ValidOrTestParm.corpus_sets[ind] for ind in range(start, end)]

    batch_xs = []
    batch_ys = []
    for record in qr_records:
        try:
            q_word_embedding = []
            q_len = len(record['q'])
            if q_len < config['max_query_len']:
                for i in range(q_len, config['max_query_len']):
                    record['q'].append(GlobalVariable.placeholder)

            if q_len > config['max_query_len']:
                record['q'] = record['q'][0: config['max_query_len']-1]

            for q_word in record['q']:
                if q_word == GlobalVariable.placeholder:
                    word_embedding = GlobalVariable.ph_embedding
                else:
                    word_embedding = GlobalVariable.wordToVector.word2vector(q_word)
                if type(word_embedding) == type(None):
                    word_embedding = GlobalVariable.ph_embedding
                q_word_embedding.append(word_embedding)

            r_word_embedding = []
            r_len = len(record['r'])
            if r_len < config['max_response_len']:
                for i in range(r_len, config['max_response_len']):
                    record['r'].append(GlobalVariable.placeholder)

            if r_len > config['max_response_len']:
                record['q'] = record['q'][0: config['max_response_len'] - 1]

            for r_word in record['r']:
                if r_word == GlobalVariable.placeholder:
                    word_embedding = GlobalVariable.ph_embedding
                else:
                    word_embedding = GlobalVariable.wordToVector.word2vector(r_word)
                if type(word_embedding) == type(None):
                    word_embedding = GlobalVariable.ph_embedding
                r_word_embedding.append(word_embedding)

            batch_xs.append(q_word_embedding)
            batch_ys.append(r_word_embedding)

        except Exception as e:
            print('corpus Error, when execute function corpus_next_batch.', "Exception: {0}".format(e))
    return batch_xs, batch_ys


def main():
    # main_operation.init_variables('English', isTraining=True)
    # for ind in range(500):
    #     batch_xs, batch_ys = get_next_batch(config['batch_size'], isValiding=True)
        # print(len(batch_xs), len(batch_ys))
        # print(len(batch_xs[0]), len(batch_ys[0]))
        # print(len(batch_xs[0][0]), len(batch_ys[0][0]))
        print()

        # if ind >= 40:
        #     break
        # break

def dataSetOpetation():
    dataSet = DataSet()
    # dataSet.dataSetPath = './corpus/QR_records_mini.json'
    # dataSet.train_rate = 0.8; dataSet.test_rate = 0.1
    # dataSet.splitDataSet('English')
    dataSet.loadDateSet('English', isTraining=True)
    print(len(dataSet.train_set))
    print(len(dataSet.valid_set))

if __name__ == '__main__':
    main()
    # dataSetOpetation()