import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from word2vec.data_reader import DataReader, Word2vecDataset
from word2vec.model import SkipGramModel

import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np


class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=3,
                 initial_lr=0.001, min_count=12, reg=None, display=False, end_of_step=None):
        self.data = DataReader(input_file, min_count)
        dataset = Word2vecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.current_iteration = 0
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.reg = reg
        self.history = {'main':[]}
        self.display = display
        self.end_of_step = end_of_step
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()
            
        self.word2vec = {}
            
    def _update_word2vec_dict(self):
        u_embeddings = self.skip_gram_model.u_embeddings.cpu()
        words = self.data.words

        for word in words:
            wid = self.data.word2id[word]
            v = u_embeddings(torch.LongTensor([wid])).detach().numpy()[0]
            self.word2vec[word] = v
            
        self.skip_gram_model.u_embeddings.cpu().to(self.device)
        
    def _display_progress(self, dots_0=150, dots_1=30):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        ax[0].title.set_text('Iteration: {}'.format(self.current_iteration + 1))
        n = len(self.history['main'])
        d = n//dots_0
        p_0 = np.zeros(dots_0)
        p_1 = np.zeros(dots_1)
        for key in self.history:
            p_0 += [np.mean(self.history[key][i*d:(i + 1)*d]) for i in range(dots_0)]
            p_1 += self.history[key][-dots_1:]
            ax[0].plot(p_0)
            ax[1].plot(p_1)
        ax[0].legend(self.history.keys())
        ax[1].legend(self.history.keys())
        plt.show()
        clear_output(True)

    def train(self):
        for iteration in range(self.current_iteration, self.iterations):
            optimizer = optim.SparseAdam(nn.ParameterList(self.skip_gram_model.parameters()), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))
            
            running_loss = 0.0
            for i, sample_batched in enumerate(self.dataloader):
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    self.history['main'].append(loss.cpu().detach())
                    if self.reg:
                        loss += self.reg(self, pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                        
                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    
                    if self.end_of_step:
                        self.end_of_step(i)
            if self.display:
                self._display_progress()
            else:
                print("Iteration: {}, Loss: {}".format(iteration, running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
            self.current_iteration += 1
        self._update_word2vec_dict()

if __name__ == '__main__':
    w2v = Word2VecTrainer(input_file="input.txt", output_file="out.vec")
    w2v.train()
