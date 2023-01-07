import re
import numpy as np
from typing import Tuple
import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, TimeDistributed, RepeatVector
from sklearn.model_selection import train_test_split

# Load Text data
# Get max english and French lengths
# (1) Tokenize lines, (2) turn to sequences and add padding
# (this step is essentially encoding the sequences)
# Build the encoder-decoder model
# Train and establish a checkpoints incase the program crashes
# Predict on test set
# Decode the predictions back into (French) words

# Needs to be completely redone from the bottom up, perhaps even split
# into more than one script

class EncodeSequences:
    def __init__(self, path_to_text:str):
        self.path = path_to_text

    def load_text(self,encoding:str='utf-8') -> np.ndarray:
        """

        Note: Complete dataset = 150,000 words, phrases and sentences
        :param encoding:
        :return: numpy array with english and French sentences split into lists
        """
        text = []
        with open(self.path, mode='r', encoding=encoding) as txt_file:
            for i, line in enumerate(txt_file):
                if i > 50000:
                    break

                split_line = line.split('\t')
                input_language = re.sub(r"[^a-zA-Z| ]",'',split_line[0]).strip()  # english
                output_language = re.sub(r"[^a-zA-Z| ]",'',split_line[1]).strip()  # French
                text.append([input_language.lower(),output_language.lower()])

        return np.asarray(text)

    @staticmethod
    def _get_sentence_lengths(text:np.array) -> Tuple[list,list]:
        """

        :param text:
        :return: input and output language sentence length lists (units=words)
        """
        in_language_lengths = [len(sequence.split()) for sequence in text[:, 0]]  # english
        out_language_lengths = [len(sequence.split()) for sequence in text[:, 1]]  # french
        return in_language_lengths, out_language_lengths

    @staticmethod
    def tokenize_and_sequence(text:np.ndarray,max_len:int,padding='post'):
        """

        :param text:
        :param max_len:
        :param padding:
        :return:
        """
        tk = Tokenizer()
        tk.fit_on_texts(text)

        sequences = tk.texts_to_sequences(text)
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding=padding)
        return padded_sequences

    def encode(self):
        pass




