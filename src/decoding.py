import _pickle as cPickle
import pandas as pd
import numpy as np

# FUNCTIONS: get_word,prediction_decoding,view_translations
class decoding():
    def __init__(self,subset_view):
        self.subset_view = subset_view

    def get_word(self,n,tk):
        for word, index in tk.word_index.items():
            if index == n:
                return word
        return None

    def prediction_decoding(self):
        # Predictions
        with open('encoded_preds.pickle','rb') as encoded_preds:
            predictions = cPickle.load(encoded_preds)
        # Tokenizers
        with open('tokenizers.pickle','rb') as tokenizers:
            eng_tokenizer, fre_tokenizer = cPickle.load(tokenizers)

        prediction_text = []
        for pred in predictions:
            temp = []
            for j in range(len(pred)):
                word = self.get_word(pred[j], eng_tokenizer)
                if j > 0:
                    if (word == self.get_word(pred[j-1], eng_tokenizer)) or (word == None): # if word==previous word or word==None
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
                
    def view_translations(self):
        with open('decoded_preds.pickle','rb') as decoded_preds:
            prediction_text = cPickle.load(decoded_preds)

        with open('data.pickle','rb') as data:
            sequence_data, test_data = cPickle.load(data)

        print('Actual Shape:{}\nPredicted Shape:{}'.format(test_data.shape,np.asarray(prediction_text).shape))

        predictions_df = pd.DataFrame({'Input French Txt':test_data[:,1], 'Predicted English Txt':prediction_text})
        pred_comparison_df = pd.DataFrame({'Actual English Txt':test_data[:,0], 'Predicted English Txt':prediction_text})

        print('HEAD:\n',predictions_df.head(self.subset_view),'\n')
        print('TAIL:\n',predictions_df.tail(self.subset_view),'\n')

        print('RANDOM COMPARISON SAMPLE:\n',pred_comparison_df.sample(self.subset_view),'\n')