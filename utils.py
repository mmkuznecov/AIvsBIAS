import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tqdm
import json

import configparser 

config = configparser.ConfigParser()
config.read('config.ini')

images_dir = config['Path']['images_dir']

class WordSurroundingsVisualizer(object):
    def __init__(self, target_word, words, surroundings_data,
                 k_neighbours=10, max_step=1, min_step=0, decrease=0.7,
                 k_vision=1.5, color='firebrick', figsize=(3, 5), right=True):
        self.target_word = target_word
        self.words = words
        self.data = surroundings_data
        self.k_neighbours = k_neighbours
        self.max_step = max_step
        self.min_step = min_step
        self.decrease = decrease
        self.k_vision = k_vision
        self.vision = max(self.k_neighbours + 1, int(self.k_neighbours*self.k_vision))
        self.color = color
        self.figsize = figsize
        self.right = right
    
    def fig_iterator(self):
        global ax
        scores = dict.fromkeys(self.words, 0)
        for record in self.data:
            for word in self.words:
                scores[word] *= self.decrease
                if word in record[0][:self.vision]:
                    word_id = record[0].index(word)
                    dist = np.linalg.norm(np.array(record[1][0]) - np.array(record[1][word_id]))
                    scores[word] += max(self.min_step, self.max_step - dist)

            selected_words = sorted(self.words, key=lambda x: -scores[x])[1:1 + self.k_neighbours]
            selected_words = selected_words[::-1]
            selected_scores = np.array([scores[x] for x in selected_words])
            
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.xaxis.set_visible(False)

            if self.right:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            
            ids = np.arange(self.k_neighbours)
            ax.hlines(y=ids, xmin=-selected_scores, xmax=0, color=self.color, alpha=0.7, linewidth=2)
            ax.scatter(x=-selected_scores, y=ids, s=75, color=self.color, alpha=1)

            ax.set_title(self.target_word, fontdict={'size':22})
            ax.set_yticks(ids)
            aligment = 'left' if self.right else 'right'
            ax.set_yticklabels(selected_words, rotation=0, fontdict={'horizontalalignment': aligment, 'size':12})
            
            yield fig, ax
            
    def display(self):
        for fig, ax in self.fig_iterator():
            plt.show()
            clear_output(True)
            
    def save(self, name, dpi=100, k=1):
        if name not in os.listdir(images_dir):
            os.mkdir(os.path.join(images_dir, name))
        plt.ioff()
        for i, (fig, ax) in tqdm(enumerate(self.fig_iterator())):
            if i % k == 0:
                fig.savefig(os.path.join(images_dir, name, '{}.jpg'.format(i//k)))
            plt.close(fig)
        plt.ion()
        
class InternalDataCollector(object):
    def __init__(self):
        self.data = {}
    
    def add_word_surroundings_data(self, data_key, trainer, target_word, k):
        trainer._update_word2vec_dict()
        target_vector = trainer.word2vec[target_word]
        words = sorted(trainer.data.words, 
                   key=lambda x: np.linalg.norm(target_vector - trainer.word2vec[x]))
        neighbors = words[:k]
        neighbors_vec = [trainer.word2vec[x].tolist() for x in neighbors]
        self.data[data_key].append([neighbors, neighbors_vec])
        
    def step_filter(self, k, f, args, step):
        if step%k == 0:
            f(*args)
            
    def splitter(self, functions, args):
        for i, f in enumerate(functions):
            f(*args[i])
    
    def add_data_key(self, key):
        self.data[key] = []
    
    def save_data(self, f):
        json.dump(self.data, f)