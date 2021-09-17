from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

# FUNCTIONS: Initializer, create_model, train_model
class model:
    def __init__(self,data,batch_size,epochs,eng_vocab_size,fre_vocab_size,e_max,f_max):
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
        model.add(Embedding(self.f_vocab,self.nodes,input_length=self.f_max,mask_zero=True))
        model.add(LSTM(self.nodes))

        model.add(RepeatVector(self.e_max))
        
        # DECODER
        model.add(LSTM(self.nodes,return_sequences=True))
        model.add(Dense(self.e_vocab,activation='softmax'))
        
        model.compile(optimizer='RMSprop',loss= 'sparse_categorical_crossentropy')
        return model

    def train_model(self):
        (x_train,y_train),(x_test,y_test) = self.data
        model = self.create_model()

        chk_point = ModelCheckpoint('best_model.tf',monitor='val_loss',verbose=1,save_best_only=True,mode='min')

        fit_model = model.fit(x_train,y_train.reshape(y_train.shape[0],y_train.shape[1],1),
            epochs=self.epochs,batch_size=self.batch_size,validation_split=.2,verbose=1,callbacks=[chk_point])
        return fit_model
