from pre_processing import pre_processing
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
        self.batch_size = 256
        self.epochs = 30
        self.preprocessing_bool = True
        self.encoding_bool = True
        self.train_bool = False
        self.predict_bool = True
        self.pred_decoding = True
        self.disp_results = True
    
    # LOADING & PRE-PROCESSING STAGE 
    def pre_processing_func(self):
        if self.preprocessing_bool == True:
            processing_obj = pre_processing(self.path,self.encoding_type,[])
            processing_obj.load_text()
            e_max_len, f_max_len = processing_obj.max_lengths()
            processing_obj.vis_lengths()

        # ENCODING STAGE 
        if self.encoding_bool == True:
            encoding_obj = encoding(self.path,self.encoding_type,[])
            encoding_obj.load_text()
            encoded_data, test, e_vocab_size, f_vocab_size, eng_tokenizer, fre_tokenizer = encoding_obj.encode()

            with open('train_test_data.pickle','wb') as data:
                cPickle.dump((encoded_data,test), data)

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
            with open('train_test_data.pickle','rb') as data:
                encoded_data, test = cPickle.load(data)

            with open('max_lengths.pickle','rb') as lengths:
                e_max_len, f_max_len = cPickle.load(lengths)

            with open('vocab_sizes.pickle','rb') as vocabs:
                e_vocab_size,f_vocab_size = cPickle.load(vocabs)

            new_model = model(encoded_data,self.batch_size,self.epochs,e_vocab_size,f_vocab_size,e_max_len,f_max_len)
            fit_model = new_model.train_model()

            # Train-Validation Loss plot
            plt.figure()
            plt.plot(fit_model.history['loss'])
            plt.plot(fit_model.history['val_loss'])
            plt.legend(['Train','Validation'])
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.savefig('training_visual.png')
            plt.show()

    # PREDICTION STAGE
    def predict_func(self):
        if self.predict_bool == True:
            best_model = load_model('best_model.tf')
            with open('train_test_data.pickle','rb') as data:
                encoded_data, test = cPickle.load(data)
            (x_train,y_train), (x_test,y_test) = encoded_data

            predictions = best_model.predict_classes(x_test.reshape((x_test.shape[0],x_test.shape[1])))

            with open('encoded_preds.pickle','wb') as encoded_preds:
                cPickle.dump(predictions,encoded_preds)

    # PREDICTION DECODING (PRED TO WORD) STAGE 
    def pred_decoding_func(self):
        if self.pred_decoding == True: 

            with open('encoded_preds.pickle','rb') as encoded_preds:
                predictions = cPickle.load(encoded_preds)

            with open('tokenizers.pickle','rb') as tokenizers:
                eng_tokenizer, fre_tokenizer = cPickle.load(tokenizers)

            with open('train_test_data.pickle','rb') as data:
                encoded_data, test = cPickle.load(data)
            (x_train,y_train), (x_test,y_test) = encoded_data

        def get_word(n, tk):
            for word, index in tk.word_index.items():
                if index == n:
                    return word
            return None

        prediction_text = []
        for pred in predictions:
            temp = []
            for j in range(len(pred)):
                word = get_word(pred[j], eng_tokenizer)
                if j > 0:
                    if (word == get_word(pred[j-1], eng_tokenizer)) or (word == None): # if word==previous word or word==None
                        temp.append('')
                    else:
                        temp.append(word)
                else:
                    if(word == None): 
                        temp.append('')
                    else:
                        temp.append(word)            
                
            prediction_text.append(' '.join(temp))

        with open('decoded_preds.pickle','wb') as decoded_preds:
            cPickle.dump(prediction_text, decoded_preds)

    # RESULTS
    def disp_results_func(self):
        if self.disp_results == True:
            with open('decoded_preds.pickle','rb') as decoded_preds:
                prediction_text = cPickle.load(decoded_preds)

            with open('train_test_data.pickle','rb') as data:
                encoded_data, test = cPickle.load(data)

            print('Actual Shape:{}\nPredicted Shape:{}'.format(test.shape,np.asarray(prediction_text).shape))

            predictions_df = pd.DataFrame({'Input French Txt':test[:,1], 'Predicted English Txt':prediction_text})
            pred_comparison_df = pd.DataFrame({'Actual English Txt':test[:,0], 'Predicted English Txt':prediction_text})

            print('HEAD:\n',predictions_df.head(15),'\n')
            print('TAIL:\n',predictions_df.tail(15),'\n')

            print('RANDOM COMPARISON SAMPLE:\n',pred_comparison_df.sample(15),'\n')