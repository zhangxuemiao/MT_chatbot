# -*- coding: utf-8 -*-
import gensim
from global_variable import GlobalVariable
import numpy as np

class WordToVector(object):
    def __init__(self, bin_file_path):
        self.bin_file_path = bin_file_path
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.bin_file_path, binary=True)


    def word2vector(self, word):
        try:
            if word in self.model.vocab:
                return self.model[word]
            else:
                return None
        except Exception as e:
            print('WordToVector Error, when retrieve word ->', word, "Details: {0}".format(e))
            return None


    def calculate_similarity(self, word1, word2):
        try:
            if word1 in self.model.vocab and word2 in self.model.vocab:
                return self.model.similarity(word1, word2)
            else:
                return None
        except Exception as e:
            print('WordToVector Error, when retrieve word ->', word1, word2, "Exception: {0}".format(e))
            return None

def to_word(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    # return words[sample]

if __name__ == '__main__':
    wordToVector = WordToVector(GlobalVariable.bin_file_path)
    # print(wordToVector.word2vector("placeholder"))

    print(len(wordToVector.model.vocab))
    for word in wordToVector.model.vocab:
        vect = wordToVector.word2vector(word)
        cumsum = np.cumsum(vect)
        sum = np.sum(vect)
        print(word, sum)
        # break