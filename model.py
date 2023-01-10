import pandas as pd
import numpy as np
from typing import Tuple
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, RepeatVector

class BuildModel:
    def __init__(self):
        pass

    def create_model(self,optimizer:str,loss:str):
        model = Sequential()

        model.add(Embedding(input_dim=1,output_dim=1,imput_length=1,mask_zero=True))
        model.add(LSTM())
        model.add(RepeatVector())
        model.add(LSTM())
        model.add(Dense(activation='relu'))

        return model.compile(optimizer=optimizer,loss=loss)

    def train(self,x_train,y_train):
        pass
