from pre_processing import pre_processing
from tensorflow.keras.models import load_model
from model import model
from encoding import encoding
import _pickle as cPickle
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

class main:
    def __init__(self):
        self.path = '../../../Datasets/fra.txt'
        self.encoding_type  = 'utf-8'
        self.batch_size = 500
        self.epochs = 30
        self.preprocessing_bool = True
        self.encoding_bool = True
        self.train_bool = False
        self.predict_bool = False
        self.pred_decoding = False
        self.disp_results = False
    
    # LOADING & PRE-PROCESSING STAGE 
    def pre_processing_func(self):
        if self.preprocessing_bool == True:
            processing_obj = pre_processing(self.path,self.encoding_type)
            processing_obj.load_and_clean()
            processing_obj.visualize_lengths()
            e_max_len, f_max_len = processing_obj.max_lengths()


        # ENCODING STAGE 
        if self.encoding_bool == True:
            encoding_obj = encoding(self.path,self.encoding_type)
            encoding_obj.load_and_clean()
            sequence_data, original_test_data, e_vocab_size, f_vocab_size, eng_tokenizer, fre_tokenizer = encoding_obj.encode()

            with open('data.pickle','wb') as data:
                cPickle.dump((sequence_data,original_test_data), data)

            with open('max_lengths.pickle','wb') as lengths:
                cPickle.dump((e_max_len,f_max_len), lengths)

            with open('vocab_sizes.pickle','wb') as vocabs:
                cPickle.dump((e_vocab_size,f_vocab_size),vocabs)

            with open('tokenizers.pickle','wb') as tokenizers:
                cPickle.dump([eng_tokenizer,fre_tokenizer],tokenizers) 

    # MODEL CREATION & TRAINING STAGE
    # CURRENT APPROXIMATE TRAIN TIME: 90 minutes
    def train_func(self):
        if self.train_bool == True:
            with open('data.pickle','rb') as data:
                sequence_data, original_test_data = cPickle.load(data)

            with open('max_lengths.pickle','rb') as lengths:
                e_max_len, f_max_len = cPickle.load(lengths)

            with open('vocab_sizes.pickle','rb') as vocabs:
                e_vocab_size,f_vocab_size = cPickle.load(vocabs)

            new_model = model(sequence_data,self.batch_size,self.epochs,e_vocab_size,f_vocab_size,e_max_len,f_max_len)
            fit_model = new_model.train_model()
            model.performance_vis(fit_model)

if __name__ == '__main__':
    main_obj = main()
    main_obj.pre_processing_func()
    main_obj.train_func()