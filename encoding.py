from pre_processing import pre_processing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences    

# FUNCTIONS: Initializer, tokenize, to_sequences, encode
class encoding(pre_processing): # pre_processing subclass
    def __init__(self,path,encoding,text):
        super().__init__(path,encoding,text)

    def tokenize(self,lines):
        tk = Tokenizer()
        tk.fit_on_texts(lines)
        return tk   

    def to_sequences(self,tokenizer,text,max_len):
        sequences = tokenizer.texts_to_sequences(text)
        sequences = pad_sequences(sequences,maxlen = max_len,padding='post')
        return sequences

    def encode(self):
        train,test = train_test_split(self.text,test_size=.15,random_state=4)

        # Tokenization
        eng_tokenizer = self.tokenize(self.text[:,0])
        fre_tokenizer = self.tokenize(self.text[:,1])

        max_eng_len,max_fre_len = self.max_lengths()

        # Seqence Encoding
        x_train = self.to_sequences(fre_tokenizer, train[:, 1], max_fre_len)
        y_train = self.to_sequences(eng_tokenizer, train[:, 0], max_eng_len)

        x_test = self.to_sequences(fre_tokenizer, test[:, 1], max_fre_len)
        y_test = self.to_sequences(eng_tokenizer, test[:, 0], max_eng_len)
        data = (x_train,y_train),(x_test,y_test)

        english_vocab_size = len(eng_tokenizer.word_index)+1
        french_vocab_size = len(fre_tokenizer.word_index)+1

        return data, test, english_vocab_size, french_vocab_size, eng_tokenizer, fre_tokenizer