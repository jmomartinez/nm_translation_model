import re
import os
import numpy as np
from typing import Tuple, Any
import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from datetime import datetime


# Load Text data
# Get max english and French lengths
# (1) Tokenize lines, (2) turn to sequences and add padding
# (this step is essentially encoding the sequences)
# Build the encoder-decoder model
# Train and establish a checkpoints incase the program crashes
# Predict on test set
# Decode the predictions back into (French) words

class EncodeSequences:
    def __init__(self, path_to_text:str):
        self.path = path_to_text
        self.text = []

    def load_text(self,encoding:str='utf-8') -> None:
        """

        Note: Complete dataset = 150,000 words, phrases and sentences
        :param encoding:
        :return: numpy array with english and French sentences split into lists
        """
        with open(self.path, mode='r', encoding=encoding) as txt_file:
            for i, line in enumerate(txt_file):
                if i > 50000:
                    break

                split_line = line.split('\t')
                input_language = re.sub(r"[^a-zA-Z| ]",'',split_line[0]).strip()  # english
                output_language = re.sub(r"[^a-zA-Z| ]",'',split_line[1]).strip()  # French
                self.text.append([input_language.lower(),output_language.lower()])

        self.text = np.asarray(self.text)

    @staticmethod
    def _max_sentence_length(text:list) -> int:
        """

        :param text:
        :return:
        """

        max_length = 0
        for sentence in text:
            if len(sentence.split()) > max_length:
                max_length = len(sentence)
        return max_length

    @staticmethod
    def _tokenize(text:list) -> Tuple[Tokenizer,int]:
        """

        :param text:
        :return:
        """

        tk = Tokenizer()
        tk.fit_on_texts(text)
        return tk, len(tk.word_index)+1

    @staticmethod
    # Method can be replaced with a TextVectorization layer as part of the model architecture
    def text_to_sequence(text_subset:np.ndarray,tokenizer:Tokenizer,
                         max_len:int,padding='post') -> Tuple[np.array,int]:
        """

        :param text_subset:
        :param tokenizer:
        :param max_len:
        :param padding:
        :return:
        """
        sequences = tokenizer.texts_to_sequences(text_subset)
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding=padding)

        return padded_sequences

    def split_data(self,test_size:float=0.20,seed:int=4) -> Any:
        """

        :param test_size:
        :param seed:
        :return:
        """
        train, test = train_test_split(self.text, test_size=test_size, random_state=seed)

        x_train, y_train = train[:,0], train[:,1]
        x_test, y_test = test[:,0], test[:,1]

        return x_train, x_test, y_train, y_test

    def process_data(self,output_tokenizers:bool=False) -> Any:
        """

        :param output_tokenizers:
        :return:
        """

        x_train, x_test, y_train, y_test = self.split_data()

        max_in_sentence_length = self._max_sentence_length(self.text[:,0])
        max_out_sentence_length = self._max_sentence_length(self.text[:,1])

        in_tokenizer, in_vocab_size = self._tokenize(self.text[:,0])
        out_tokenizer, out_vocab_size = self._tokenize(self.text[:,1])

        x_train_seqs = self.text_to_sequence(x_train,in_tokenizer,max_in_sentence_length)
        y_train_seqs = self.text_to_sequence(y_train,out_tokenizer,max_out_sentence_length)

        x_test_seqs = self.text_to_sequence(x_test,in_tokenizer,max_in_sentence_length)
        y_test_seqs = self.text_to_sequence(y_test,out_tokenizer,max_out_sentence_length)

        if output_tokenizers:
            self.dump_pickles(in_tokenizer=in_tokenizer,out_tokenizer=out_tokenizer)

        return x_train_seqs, x_test_seqs, y_train_seqs, y_test_seqs

    @staticmethod
    def dump_pickles(**kwargs) -> None:
        """

        :param kwargs:
        :return:
        """
        now = datetime.now().strftime("%m-%d-%Y %H-%M-%S")

        for key, val in kwargs.items():
            os.makedirs(os.path.dirname('output_files/'), exist_ok=True)
            path = f'output_files/{key}_{now}.pkl'
            with open(path,'wb') as file:
                pickle.dump(kwargs[key],file)
















