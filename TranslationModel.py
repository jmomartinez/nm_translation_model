import os
import numpy as np
from typing import Any
from keras.callbacks import ModelCheckpoint, History
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from datetime import datetime
from ModelArgs import ModelArgs


class TranslationModel:
    def __init__(self, model_args: ModelArgs, x_train: np.ndarray, y_train: np.ndarray):
        self.args = model_args
        self.x_train = x_train
        self.y_train = y_train

    def create_model(self) -> Sequential:
        """
        Creates encoder-decoder model architecture according to the arg specification and compiles the model
        The use of LSTM cells here increases the model's robustness for longer sequences and avoids gradient problems
        A future state of this model will likely include a transformer architecture with an attention mechanism
        :return: Compiled model object
        """
        model = Sequential()

        model.add(Embedding(input_dim=self.args.in_vocab_size, output_dim=self.args.vector_space_size,
                            input_length=self.args.max_in_length, mask_zero=True))
        model.add(LSTM(self.args.vector_space_size))
        model.add(RepeatVector(self.args.max_out_length))
        model.add(LSTM(self.args.vector_space_size, return_sequences=True))
        model.add(Dense(units=self.args.out_vocab_size, activation='softmax'))
        model.compile(optimizer=self.args.optimizer, loss=self.args.loss_function)

        return model

    def train(self, model: Sequential) -> History:
        """
        Trains translation model with a check point callback based on the validation loss to avoid overfitting
        :param model: compiled model object
        :return: trained model
        """
        self.y_train = self.y_train.reshape(self.y_train.shape[0], self.y_train.shape[1], 1)
        chk_point = ModelCheckpoint('best_model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        if self.args.validation_split:
            history = model.fit(self.x_train, self.y_train.reshape(self.y_train.shape[0], self.y_train.shape[1], 1), epochs=self.args.epochs, batch_size=self.args.batch_size,
                                validation_split=self.args.validation_split, verbose=2, callbacks=[chk_point])
        else:
            history = model.fit(self.x_train, self.y_train.reshape(self.y_train.shape[0], self.y_train.shape[1], 1), epochs=self.args.epochs, batch_size=self.args.batch_size,
                                verbose=2,callbacks=[chk_point])

        self._output_model(model)
        return history

    @staticmethod
    def _output_model(fit_model: Any) -> None:
        """
        Dumps model pickle file
        :return: None. Makes directory if not present and dumps pkl file
        """
        now = datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        os.makedirs(os.path.dirname('output_files/'), exist_ok=True)
        path = f'output_files/fit_nmt_model_{now}.keras'
        fit_model.save(path, overwrite=False, save_format='keras')
