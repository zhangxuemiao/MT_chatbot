# coding: utf-8

import glob
import json
import nltk
from nltk.corpus import stopwords
import sys, os, chardet


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

    # def WordCheck(self,words):#拼写检查
    #     d = enchant.Dict("en_US")
    #     checkedWords=()
    #     for word in words:
    #         if not d.check(word):
    #             d.suggest(word)
    #             word=raw_input()
    #         checkedWords = (checkedWords,'05')
    #     return checkedWords
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

def ubuntu_qr_records_gen(raw_corpus_path, corpus_save_dir):
    # file_dir = GlobalVariable.ubuntu_dialogs_path
    file_dir = raw_corpus_path
    standard_file_name = "/*/*.tsv"
    sentence2Words = Sentence2Words()
    with open(corpus_save_dir+'/ubuntu_qr_records.json', 'w') as json_records:
        record_count = 0
        # exit_flag = False
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
                            if record_count % 10000 == 0:
                                print('Has store ', record_count, 'query-response records')

                            # generate mini corpus when needing
                            # if record_count >= 9000:
                            #     exit_flag = True
                            #     break

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
            # if exit_flag:
            #     break


def weibo_qr_records_gen(raw_corpus_path_q, raw_corpus_path_r, corpus_save_dir):
    print(corpus_save_dir+'/weibo_qr_records.json')
    with open(corpus_save_dir+'/weibo_qr_records.json', 'w', encoding='utf-8') as json_records:
        record_count = 0
        raw_q_list = []
        raw_r_list = []
        with open(raw_corpus_path_q, 'r', encoding='utf-8') as raw_q:
            for line in raw_q.readlines():
                raw_q_list.append(line)
        with open(raw_corpus_path_r, 'r', encoding='utf-8') as raw_r:
            for l in raw_r.readlines():
                raw_r_list.append(l)

        min_len = len(raw_r_list)
        if len(raw_q_list) <= len(raw_r_list):
            min_len = len(raw_q_list)
        QR_record = {"q": None, 'r': None}
        for i in range(min_len):
            QR_record['q'] = raw_q_list[i]
            QR_record['r'] = raw_r_list[i]

            # print(QR_record)
            json_records.write(json.dumps(QR_record, ensure_ascii=False) + '\n')
            print(json.dumps(QR_record, ensure_ascii=False))
            record_count += 1
            if record_count % 10000 == 0:
                print('Has store ', record_count, 'query-response records')

def test_weibo():
    with open("./corpus/stc_weibo_train_post_mini.txt", "rb") as f:
        data = f.read()
        print(chardet.detect(data))
    # with open('./corpus/weibo_qr_records.json', 'r', encoding='utf-8') as f:
    #     for i in f.readlines():
    #         print(i)


def japanese_qr_records_gen(corpus_file_path):

    pass


def common_corpus_gen(corpus_file_path):

    pass


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
        print(len(corpus))
        return corpus


def sentence_clean(sentence_):
    sentence = """hi, i installed xubuntu on my laptop. i cant use wifi because it says "wireless disabled by hardware switch" -  i tried rfkill ublock as suggested in many forums, but i does not help. i have a w-lan button, but it is not recognized by windows. what can i do?\n"""
    if len(sentence) <= 0:
        return None
    sentence2Words = Sentence2Words()
    print(sentence2Words.standard_operation(sentence))

def ubuntu_dialog_main():
    if len(sys.argv) <5:
        print("Usage: python corpus_prepare.py -c raw_corpus_dir -o corpus_save_dir -i class")
        print("Such as: python corpus_prepare.py -c /tmp/ubuntu_dialogs/dialogs -o ./corpus -m ubuntu_dialog")
        sys.exit(-1)
    if '-c' != sys.argv[1] or '-o' != sys.argv[3]:
        print("Usage: python corpus_prepare.py -c raw_corpus_dir -o corpus_save_dir")
        print("Such as: python corpus_prepare.py -c /tmp/ubuntu_dialogs/dialogs -o ./corpus")
        sys.exit(-1)

    raw_corpus_dir = sys.argv[2]
    corpus_save_dir = sys.argv[4]
    if not os.path.exists(corpus_save_dir):
        os.mkdir(corpus_save_dir)

    try:
        ubuntu_qr_records_gen(raw_corpus_dir, corpus_save_dir)
    except Exception as e:
        print('Exceptions occurs when generate corpus from', raw_corpus_dir, 'to', corpus_save_dir,', details like this:', "Exception: {0}".format(e))
def weibo_main():
    if len(sys.argv) <5:
        print("Usage: python corpus_prepare.py -c raw_q_corpus raw_r_corpus -o corpus_save_dir -i class")
        print("Such as: python corpus_prepare.py -c ./q ./r -o ./corpus -m ubuntu_dialog")
        sys.exit(-1)
    if '-c' != sys.argv[1] or '-o' != sys.argv[4]:
        print("Usage: python corpus_prepare.py -c raw_corpus_dir -o corpus_save_dir")
        print("Such as: python corpus_prepare.py -c /tmp/ubuntu_dialogs/dialogs -o ./corpus")
        sys.exit(-1)

    raw_q = sys.argv[2]
    raw_r = sys.argv[3]
    corpus_save_dir = sys.argv[5]
    if not os.path.exists(corpus_save_dir):
        os.mkdir(corpus_save_dir)

    try:
        weibo_qr_records_gen(raw_q, raw_r, corpus_save_dir)
    except Exception as e:
        print('Exceptions occurs when generate corpus from', raw_q, raw_r, 'to', corpus_save_dir,', details like this:', "Exception: {0}".format(e))

def main():
    weibo_main()
    # test_weibo()
    pass

if __name__ == '__main__':
    main()