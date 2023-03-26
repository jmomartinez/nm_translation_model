from keras.models import load_model
import numpy as np


class SequenceDecoding:
    def __init__(self,model_path):
        self.model_path = model_path

    def predict_translation(self,x_test:np.ndarray):
        """

        :param x_test:
        :return:
        """
        fit_model = load_model(self.model_path)
        predictions = fit_model.predict_classes(x_test)

    @staticmethod
    def get_word(thn,sklsd):

        pass

    def decode_predictions(self):
        pass

