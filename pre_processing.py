import numpy as np
import pandas as pd
import re

# FUNCTIONS: Initializer, load_text, text_length, max_lengths, vis_lengths
class pre_processing: # Parent Class
    def __init__(self,path,encoding,text):
        self.path = path
        self.encoding = encoding
        self.text = text

    def load_text(self):
        text = []
        for line in open(self.path,mode = 'r', encoding = self.encoding):
                line = line.strip().split('\t')
                line = '|'.join(line[:1]+line[1:2])
                line = re.sub(r"[^a-z A-Z|]",'',line).lower().split('|')
                text.append(line)

        text = np.asarray(text)
        self.text = text[:50000,:] # Complete dataset = 150,000 words, phrases and sentences

    def text_lengths(self):
        e_lengths = []
        f_lengths = []

        for i in self.text[:,0]:
            e_lengths.append(len(i.split()))

        for i in self.text[:,1]:
            f_lengths.append(len(i.split()))

        return e_lengths,f_lengths

    def max_lengths(self):
        e_lengths,f_lengths = self.text_lengths()
        return max(e_lengths), max(f_lengths)

    def vis_lengths(self):
        e_lengths, f_lengths = self.text_lengths()
        df = pd.DataFrame({'English':e_lengths, 'French':f_lengths}).hist(bins=25)
        # x-axis = Sentence Length 
        # y-axis Sentence Instances
        # plt.show()