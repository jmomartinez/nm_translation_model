import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import json

# FUNCTIONS: Constructor, load_and_clean,text_length, max_lengths, visualize_lengths
class pre_processing: # Parent Class
    def __init__(self,path,encoding):
        self.path = path
        self.encoding = encoding
        self.text = 0
        self.e_lengths = 0
        self.f_lengths = 0

    def load_and_clean(self):
        text = []
        for line in open(self.path,mode = 'r', encoding = self.encoding):
                line = line.split('\t')
                line = '|'.join(line[:1]+line[1:2])
                line = re.sub(r"[^a-z A-Z|]",'',line).lower().split('|')
                text.append(line)

        text = np.asarray(text)
        self.text = text[:50000,:] # Complete dataset = 150,000 words, phrases and sentences

    def text_lengths(self):
        e_lengths = [len(seq1.split()) for seq1 in self.text[:,0]]
        f_lengths = [len(seq2.split()) for seq2 in self.text[:,1]]

        self.e_lengths,self.f_lengths = e_lengths,f_lengths

    def visualize_lengths(self):
        self.load_and_clean()
        self.text_lengths()
        pd.DataFrame({'English':self.e_lengths, 'French':self.f_lengths}).hist(bins=20)
        plt.savefig('Sequence Length Distribution.png')
        plt.show()
        # x-axis = Sequence Length 
        # y-axis Sequence length Instances (i.e. # of times a sequence of length n appears)

    def max_lengths(self):
        if self.text==0:
            self.load_and_clean()
        if self.e_lengths == 0:
            self.text_lengths()
        return max(self.e_lengths), max(self.f_lengths)