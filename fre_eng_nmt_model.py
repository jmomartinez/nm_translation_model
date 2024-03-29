from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, TimeDistributed, RepeatVector
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import ModelCheckpoint
import _pickle as cPickle
import pandas as pd
import numpy as np
import re
from typing import Tuple


# FUNCTIONS: Initializer, load_text, text_length, max_lengths, vis_lengths
class PreProcessing:  # Parent Class
    def __init__(self, path: str, encoding: str):
        self.path = path
        self.encoding = encoding
        self.text = []

    def load_text(self) -> None:
        ''' 

        Note: Complete dataset = 150,000 words, phrases and sentences
        args:

        output:

        '''
        with open(self.path, mode='r', encoding=self.encoding) as txt_file:
            for i, line in enumerate(txt_file):
                if i > 50000:
                    break
                split_line = line.strip().split('\t')
                eng = re.sub(r"[^a-zA-Z]", '', split_line[0]).lower()
                fre = re.sub(r"[^a-zA-Z]", '', line_list[1]).lower()
                self.text.append([eng, fre])

        self.text = np.asarray(self.text)

    def sequence_lengths(self) -> Tuple[list, list]:
        ''' 

        args:

        output:

        '''
        english_lengths = [len(sequence.split()) for sequence in self.text[:, 0]]
        french_lengths = [len(sequence.split()) for sequence in self.text[:, 1]]

        return english_lengths, french_lengths, max(eng_lengths), max(fre_lengths)

    # def vis_lengths(self):
    #     e_lengths, f_lengths = self.text_lengths()
    #     df = pd.DataFrame({'English':e_lengths, 'French':f_lengths}).hist(bins=25)
    #     # x-axis = Sentence Length 
    #     # y-axis Sentence Instances
    #     plt.show()


# FUNCTIONS: Initializer, tokenize, to_sequences, encode
class encoding(PreProcessing):  # PreProcessing subclass
    def __init__(self, path, encoding, text):
        super().__init__(path, encoding, text)

    def tokenize(self, lines) -> object:
        """

        :return:
        :param lines:
        :return:
        """
        tk = Tokenizer()
        tk.fit_on_texts(lines)
        return tk

    def to_sequences(self, tokenizer, text, max_len):
        """

        :param tokenizer:
        :param text:
        :param max_len:
        :return:
        """
        sequences = tokenizer.texts_to_sequences(text)
        sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
        return sequences

    def encode(self):
        train, test = train_test_split(self.text, test_size=.15, random_state=4)

        # Tokenization
        eng_tokenizer = self.tokenize(self.text[:, 0])
        fre_tokenizer = self.tokenize(self.text[:, 1])

        max_eng_len, max_fre_len = self.max_lengths()

        # Sequence Encoding
        x_train = self.to_sequences(fre_tokenizer, train[:, 1], max_fre_len)
        y_train = self.to_sequences(eng_tokenizer, train[:, 0], max_eng_len)

        x_test = self.to_sequences(fre_tokenizer, test[:, 1], max_fre_len)
        y_test = self.to_sequences(eng_tokenizer, test[:, 0], max_eng_len)
        data = (x_train, y_train), (x_test, y_test)

        english_vocab_size = len(eng_tokenizer.word_index) + 1
        french_vocab_size = len(fre_tokenizer.word_index) + 1

        return data, test, english_vocab_size, french_vocab_size, eng_tokenizer, fre_tokenizer


# FUNCTIONS: Initializer, create_model, train_model
class model:
    def __init__(self, data, batch_size, epochs, eng_vocab_size, fre_vocab_size, e_max, f_max):
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs
        self.e_vocab = eng_vocab_size
        self.f_vocab = fre_vocab_size
        self.e_max = e_max
        self.f_max = f_max
        self.nodes = 512

    def create_model(self):
        model = Sequential()
        # ENCODER
        model.add(Embedding(self.f_vocab, self.nodes, input_length=self.f_max, mask_zero=True))
        model.add(LSTM(self.nodes))

        model.add(RepeatVector(self.e_max))

        # DECODER
        model.add(LSTM(self.nodes, return_sequences=True))
        model.add(Dense(self.e_vocab, activation='softmax'))

        model.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy')
        return model

    def train_model(self):
        (x_train, y_train), (x_test, y_test) = self.data
        model = self.create_model()

        chk_point = ModelCheckpoint('best_model.tf', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        fit_model = model.fit(x_train, y_train.reshape(y_train.shape[0], y_train.shape[1], 1),
                              epochs=self.epochs, batch_size=self.batch_size, validation_split=.2, verbose=1,
                              callbacks=[chk_point])
        return fit_model


class main:
    def __init__(self):
        self.path = 'fra.txt'
        self.encoding_type = 'utf-8'
        self.batch_size = 256
        self.epochs = 30
        self.preprocessing_bool = True
        self.encoding_bool = True
        self.train_bool = False
        self.predict_bool = True
        self.pred_decoding = True
        self.disp_results = True

    # LOADING & PRE-PROCESSING STAGE 
    def PreProcessing_func(self):
        if self.preprocessing_bool == True:
            processing_obj = PreProcessing(self.path, self.encoding_type, [])
            processing_obj.load_text()
            e_max_len, f_max_len = processing_obj.max_lengths()
            processing_obj.vis_lengths()

        # ENCODING STAGE 
        if self.encoding_bool == True:
            encoding_obj = encoding(self.path, self.encoding_type, [])
            encoding_obj.load_text()
            encoded_data, test, e_vocab_size, f_vocab_size, eng_tokenizer, fre_tokenizer = encoding_obj.encode()

            with open('train_test_data.pickle', 'wb') as data:
                cPickle.dump((encoded_data, test), data)

            with open('max_lengths.pickle', 'wb') as lengths:
                cPickle.dump((e_max_len, f_max_len), lengths)

            with open('vocab_sizes.pickle', 'wb') as vocabs:
                cPickle.dump((e_vocab_size, f_vocab_size), vocabs)

            with open('tokenizers.pickle', 'wb') as tokenizers:
                cPickle.dump([eng_tokenizer, fre_tokenizer], tokenizers)

                # MODEL CREATION & TRAINING STAGE

    # CURRENT APPROXIMATE TRAIN TIME: 90 minutes
    def train_func(self):
        if self.train_bool == True:
            with open('train_test_data.pickle', 'rb') as data:
                encoded_data, test = cPickle.load(data)

            with open('max_lengths.pickle', 'rb') as lengths:
                e_max_len, f_max_len = cPickle.load(lengths)

            with open('vocab_sizes.pickle', 'rb') as vocabs:
                e_vocab_size, f_vocab_size = cPickle.load(vocabs)

            new_model = model(encoded_data, self.batch_size, self.epochs, e_vocab_size, f_vocab_size, e_max_len,
                              f_max_len)
            fit_model = new_model.train_model()

            # Train-Validation Loss plot
            plt.figure()
            plt.plot(fit_model.history['loss'])
            plt.plot(fit_model.history['val_loss'])
            plt.legend(['Train', 'Validation'])
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.savefig('training_visual.png')
            plt.show()

    # PREDICTION STAGE
    def predict_func(self):
        if self.predict_bool == True:
            best_model = load_model('best_model.tf')
            with open('train_test_data.pickle', 'rb') as data:
                encoded_data, test = cPickle.load(data)
            (x_train, y_train), (x_test, y_test) = encoded_data

            # These predictions are sequences of integers.
            # We need to convert these integers to their corresponding words.
            predictions = best_model.predict_classes(x_test.reshape((x_test.shape[0], x_test.shape[1])))

            with open('encoded_preds.pickle', 'wb') as encoded_preds:
                cPickle.dump(predictions, encoded_preds)

    # PREDICTION DECODING (PRED TO WORD) STAGE 
    def pred_decoding_func(self):
        if self.pred_decoding:
            with open('encoded_preds.pickle', 'rb') as encoded_preds:
                predictions = cPickle.load(encoded_preds)

            with open('tokenizers.pickle', 'rb') as tokenizers:
                eng_tokenizer, fre_tokenizer = cPickle.load(tokenizers)

            with open('train_test_data.pickle', 'rb') as data:
                encoded_data, test = cPickle.load(data)
            (x_train, y_train), (x_test, y_test) = encoded_data

        def get_word(word_index, tk):
            for word, index in tk.word_index.items():
                if index == word_index:
                    return word
            return None

        def decode_predictions():
        text_predictions = []
        for pred in predictions:
            current_prediction = []
            for i in range(len(pred)):
                word = get_word(pred[i], eng_tokenizer)
                if i == 0 and word:
                    current_prediction.append(word)
                elif i > 0 and word != get_word(pred[i-1], eng_tokenizer) and word:
                    current_prediction.append(word)

            text_predictions.append(' '.join(current_prediction))




        prediction_text = []
        for pred in predictions:
            temp = []
            for j in range(len(pred)):
                word = get_word(pred[j], eng_tokenizer)
                # checks if it's a single word/index, if so just append it, otherwise check do the check below
                if j > 0:
                    # if word==previous word or word==None
                    '''essentially checks to see if the current word is equal to the previous word or if ts None
                    if so then an empty string is appended, otherwise if they're unique then you append the current
                    word'''
                    if word == get_word(pred[j - 1], eng_tokenizer) or not word:
                        temp.append('')
                    else:
                        temp.append(word)
                else:
                    if not word:
                        temp.append('')
                    else:
                        temp.append(word)

            prediction_text.append(' '.join(temp))

        with open('decoded_preds.pickle', 'wb') as decoded_preds:
            cPickle.dump(prediction_text, decoded_preds)

    # RESULTS
    def disp_results_func(self):
        if self.disp_results == True:
            with open('decoded_preds.pickle', 'rb') as decoded_preds:
                prediction_text = cPickle.load(decoded_preds)

            with open('train_test_data.pickle', 'rb') as data:
                encoded_data, test = cPickle.load(data)

            print('Actual Shape:{}\nPredicted Shape:{}'.format(test.shape, np.asarray(prediction_text).shape))

            predictions_df = pd.DataFrame({'Input French Txt': test[:, 1], 'Predicted English Txt': prediction_text})
            pred_comparison_df = pd.DataFrame(
                {'Actual English Txt': test[:, 0], 'Predicted English Txt': prediction_text})

            print('HEAD:\n', predictions_df.head(15), '\n')
            print('TAIL:\n', predictions_df.tail(15), '\n')

            print('RANDOM COMPARISON SAMPLE:\n', pred_comparison_df.sample(15), '\n')


if __name__ == '__main__':
    main_obj = main()
    main_obj.PreProcessing_func()
    main_obj.train_func()
    main_obj.predict_func()
    main_obj.pred_decoding_func()
    main_obj.disp_results_func()
