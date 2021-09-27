from pre_processing import pre_processing
from encoding import encoding
from model import model
from decoding import decoding
import _pickle as cPickle

# FUNCTIONS: Constructor,pre_process,encode_data,train_predict,decode_preds
class main:
    def __init__(self,run_opts):
        self.run_opts = run_opts
        self.path = '../fra.txt'
        self.encoding_type  = 'utf-8'

    def pre_process(self):
        if self.run_opts['dpp']==True:
            processing_obj = pre_processing(self.path,self.encoding_type)
            processing_obj.visualize_lengths()

    def encode_data(self):
        if self.run_opts['encode']==True:
            encoding_obj = encoding(self.path,self.encoding_type)
            e_max_len, f_max_len = encoding_obj.max_lengths()
            
            sequence_data, original_test_data, e_vocab_size, f_vocab_size, eng_tokenizer, fre_tokenizer = encoding_obj.encode()
            #Dumping data,max_lens,vocab_sizes and tokenizer objects
            with open('data.pickle','wb') as data:
                cPickle.dump((sequence_data,original_test_data), data)
            with open('max_lengths.pickle','wb') as lengths:
                cPickle.dump((e_max_len,f_max_len), lengths)
            with open('vocab_sizes.pickle','wb') as vocabs:
                cPickle.dump((e_vocab_size,f_vocab_size),vocabs)
            with open('tokenizers.pickle','wb') as tokenizers:
                cPickle.dump([eng_tokenizer,fre_tokenizer],tokenizers) 

    def train_predict(self):
        if self.run_opts['train_predict']==True:
            batch_size,epochs = 250,30
            #Loading data,max_lens and vocab sizes
            with open('data.pickle','rb') as data:
                sequence_data, original_test_data = cPickle.load(data)
            with open('max_lengths.pickle','rb') as lengths:
                e_max_len, f_max_len = cPickle.load(lengths)
            with open('vocab_sizes.pickle','rb') as vocabs:
                e_vocab_size,f_vocab_size = cPickle.load(vocabs)

            new_model = model(sequence_data,batch_size,epochs,e_vocab_size,f_vocab_size,e_max_len,f_max_len)
            fit_model = new_model.train_model()
            new_model.performance_vis(fit_model)
            new_model.predict()

    def decode_preds(self):
        if self.run_opts['decode']==True:
            subset_view = 15
            decoding_obj = decoding(subset_view)
            decoding_obj.prediction_decoding()
            decoding_obj.view_translations()

if __name__ == '__main__':
    run_opts = {'dpp':True,'encode':True,'train_predict':True,'decode':True}
    main_obj = main(run_opts)
    main_obj.pre_process()
    main_obj.encode_data()
    main_obj.train_predict()
    main_obj.decode_preds()