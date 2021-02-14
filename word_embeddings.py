import json
import numpy as np

import configparser 

config = configparser.ConfigParser()
config.read('config.ini')

female_words  = config['GenderSpecificWords']['female_words'].split(', ')
male_words    = config['GenderSpecificWords']['male_words'].split(', ')
output_dir    = config['Path']['output_dir']

class WordEmbeddings(object):
    def __init__(self, word2id=None, word2vec=None):
        self.mean_males = None
        self.mean_females = None
        self.filtered_words = None
        if word2id and word2vec:
            self.word2id = word2id
            self.id2word = dict((v,k) for k,v in word2id.items())
            self.word2vec = word2vec
            self.words = list(word2id.keys())
            self.vectors = list(word2vec.values())
            self.update()
        else:
            self.word2id = {}
            self.id2word = {}
            self.word2vec = {}
            self.words = []
            self.vectors = []
        
    def update(self):
        vec_males = np.array([self.word2vec[x] for x in male_words if x in self.words])
        self.mean_males = np.mean(vec_males, 0)

        vec_females = np.array([self.word2vec[x] for x in female_words if x in self.words])
        self.mean_females = np.mean(vec_females, 0)
        
        self.filtered_words = [x for x in self.words if (x not in female_words and x not in male_words)]
        
    def w_diff(self, word):
        m_dist = np.linalg.norm(self.mean_males - self.word2vec[word])
        f_dist = np.linalg.norm(self.mean_females - self.word2vec[word])
        return m_dist - f_dist
    
    def m_diff(self, word):
        m_dist = np.linalg.norm(self.mean_males - self.word2vec[word])
        f_dist = np.linalg.norm(self.mean_females - self.word2vec[word])
        return f_dist - m_dist
    
    def f_dist(self, word):
        return np.linalg.norm(self.mean_females - self.word2vec[word])
    def m_dist(self, word):
        return np.linalg.norm(self.mean_males - self.word2vec[word])

    def save(self, path):
        word2vec = {}
        for word in self.words:
            word2vec[word] = self.word2vec[word].tolist()
        with open(path, 'w') as f:
            json.dump(word2vec, f)
    
    def load(self, path):
        with open(path, 'r') as f:
            self.word2vec = json.load(f)
        self.words = list(self.word2vec.keys())
        self.id2word = dict(zip([i for i in range(len(self.words))], self.words))
        self.word2id = dict((v,k) for k,v in self.id2word.items())
        self.update()