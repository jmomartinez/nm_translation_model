import re
import os
import pickle
import numpy as np
from typing import Tuple, Any, Dict
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from datetime import datetime
from keras.models import load_model
from typing import Union, List


class SequenceEncoding:
    def __init__(self, path_to_text: str, line_limit: int = None):
        self.path = path_to_text
        self.line_limit = line_limit
        self.text = []

    def load_text(self, encoding: str = 'utf-8') -> None:
        """
        Loads corpus line by line from a .txt file up to the limit, if specified, otherwise loads the entire text
        Removes anything that isn't a letter and splits language pairs into lists
        Note: Complete dataset = 150,000 words, phrases and sentences (French to english)
        Link to datasets: http://www.manythings.org/anki/
        :param encoding: (optional) utf-8 by default, but can be specified
        :return: None. self.text is updated as a numpy array with input-output language pairs in lists
        """
        with open(self.path, mode='r', encoding=encoding) as reader:
            for i, line in enumerate(reader):
                if not self.line_limit:
                    pass
                else:
                    if i > self.line_limit:
                        break

                split_line = line.split('\t')

                # TODO: Find a way to express apostrophes and important punctuation in sequences
                input_language = re.sub(r"[^a-zA-Z| ]", '', split_line[0]).strip()  # english
                output_language = re.sub(r"[^a-zA-Z| ]", '', split_line[1]).strip()  # French
                self.text.append([input_language.lower(), output_language.lower()])

        self.text = np.asarray(self.text)

    @staticmethod
    def _max_sentence_length(text: list) -> int:
        """
        Finds the longest sentence length for a language subset of the text (units=words)
        :param text: text subset (slice) containing sentence lists for a single language
        :return: the max sentence length
        """
        max_length = 0
        for sentence in text:
            if len(sentence.split()) > max_length:
                max_length = len(sentence)
        return max_length

    @staticmethod
    def _tokenize(text: list) -> Tuple[Tokenizer, int]:
        """
        Fits a Tokenizer obj on a language subset of the text (i.e. only english)
        :param text: text subset (slice) containing sentence lists for a single language
        :return: fit Tokenizer obj and the text's vocab size (num of unique words)
        """
        tk = Tokenizer()
        tk.fit_on_texts(text)
        return tk, len(tk.word_index) + 1

    @staticmethod
    # Method can be replaced with a TextVectorization layer as part of the model architecture
    def text_to_sequence(text_subset: np.ndarray, tokenizer: Tokenizer,
                         max_len: int, padding: str = 'post') -> np.ndarray:
        """
        Transforms the texts into sequences based on the index dictionary the Tokenizer object
        created when it was fit on the text subset; pads sequences with zeros to match the longest sequence
        :param text_subset:
        :param tokenizer: trained Tokenizer() obj
        :param max_len: max sentence length found in text for a particular language
        :param padding: (optional) positioning of the zero padding
        :return: array of padded sequences
        """
        sequences = tokenizer.texts_to_sequences(text_subset)
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding=padding)

        return padded_sequences

    def split_data(self, test_size: float = 0.20, seed: int = 4) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits text data into train and test where x and y represent the input and output languages, respectively
        :param test_size: (optional) percentage of data to save for testing
        :param seed: seed for reproducibility
        :return: train and test subsets
        """
        train, test = train_test_split(self.text, test_size=test_size, random_state=seed)

        x_train, y_train = train[:, 0], train[:, 1]
        x_test, y_test = test[:, 0], test[:, 1]

        return x_train, x_test, y_train, y_test

    def get_vocab_metadata(self) -> Dict[str, Any]:
        """
        Calls _tokenize() and _max_sentence_length() to form a vocab metadata dictionary containing
        input and output language details (max len, vocab size, and fitted Tokenizer() obj)
        :return: vocab metadata dict
        """
        max_in_length = self._max_sentence_length(self.text[:, 0])
        max_out_length = self._max_sentence_length(self.text[:, 1])

        print(f'Max IN: {max_in_length}, Max OUT: {max_out_length}')

        in_tokenizer, in_vocab_size = self._tokenize(self.text[:, 0])
        out_tokenizer, out_vocab_size = self._tokenize(self.text[:, 1])

        return {'max_in_length': max_in_length, 'in_vocab_size': in_vocab_size, 'in_tok': in_tokenizer,
                'out_vocab_size': out_vocab_size, 'max_out_length': max_out_length, 'out_tok': out_tokenizer}

    def process_data(self, vocab_metadata: dict, output_tokenizers: bool = False) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes data by calling relevant functions to split text into train and test subsets and
        convert texts into padded sequences; optionally pickles Tokenizer() objects
        :param vocab_metadata: dict with metadata needed to convert texts to sequences
        :param output_tokenizers: optional boolean
        :return: train and test subsets as transformed sequences for model training
        """
        x_train, x_test, y_train, y_test = self.split_data()

        x_train_seqs = self.text_to_sequence(x_train, vocab_metadata['in_tok'], vocab_metadata['max_in_length'])
        y_train_seqs = self.text_to_sequence(y_train, vocab_metadata['out_tok'], vocab_metadata['max_out_length'])

        x_test_seqs = self.text_to_sequence(x_test, vocab_metadata['in_tok'], vocab_metadata['max_in_length'])
        y_test_seqs = self.text_to_sequence(y_test, vocab_metadata['out_tok'], vocab_metadata['max_out_length'])

        if output_tokenizers:
            self._pickle_tokenizers(in_tokenizer=vocab_metadata['in_tok'], out_tokenizer=vocab_metadata['out_tok'])

        return x_train_seqs, x_test_seqs, y_train_seqs, y_test_seqs

    @staticmethod
    def _pickle_tokenizers(in_tokenizer: Tokenizer, out_tokenizer: Tokenizer) -> None:
        """
        Dumps input and output tokenizers as pickle files
        :return: None. Makes dir if not present and dumps pkl files
        """
        directory = 'output_files'
        os.makedirs(directory, exist_ok=True)
        now = datetime.now().strftime("%m-%d-%Y %H-%M-%S")

        with open(f'{directory}/in_tokenizer_{now}.pkl', 'wb') as out_file1:
            pickle.dump(in_tokenizer, out_file1)

        with open(f'{directory}/out_tokenizer_{now}.pkl', 'wb') as out_file2:
            pickle.dump(out_tokenizer, out_file2)


class SequenceDecoding:
    def __init__(self, trained_model_path: str, output_language_tokenizer: Tokenizer):
        self.model_path = trained_model_path
        self.tokenizer = output_language_tokenizer

    def predict_translation(self, x_test: np.ndarray) -> np.ndarray:
        fit_model = load_model(self.model_path)
        return fit_model.predict(x_test.reshape((x_test.shape[0], x_test.shape[1])))

    def get_word(self, word_index) -> Union[None, str]:
        for word, index in self.tokenizer.word_index.items():
            print(word,index,word_index)
            if index == word_index:
                return word
        return None

    def decode_predictions(self, predictions) -> List[str]:
        decoded_preds = []

        for pred in predictions:
            current_prediction = []
            for i in range(len(pred)):
                word = self.get_word(pred[i])
                if i == 0 and word:
                    current_prediction.append(word)
                # I question the need to check for word uniqueness, don't think it happens often
                elif i > 0 and word != self.get_word(pred[i - 1]) and word:
                    current_prediction.append(word)

            decoded_preds.append(' '.join(current_prediction))

        return decoded_preds
