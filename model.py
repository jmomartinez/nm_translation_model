from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
import _pickle as cPickle

# FUNCTIONS: Initializer, create_model, train_model
class model:
    def __init__(self,data,batch_size,epochs,eng_vocab_size,fre_vocab_size,e_max,f_max):
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs
        self.eng_vocab_size = eng_vocab_size
        self.fre_vocab_size = fre_vocab_size
        self.e_max = e_max
        self.f_max = f_max
        self.embedding_vectors = 500

    def create_model(self):
        model = Sequential()
        # ENCODER
        model.add(Embedding(self.fre_vocab_size,self.embedding_vectors,input_length=self.f_max,mask_zero=True))
        model.add(LSTM(self.embedding_vectors))

        model.add(RepeatVector(self.e_max))
        
        # DECODER
        model.add(LSTM(self.embedding_vectors,return_sequences=True))
        model.add(Dense(self.eng_vocab_size,activation='softmax'))
        
        model.compile(optimizer='RMSprop',loss= 'sparse_categorical_crossentropy',metrics=['accuracy'])
        return model
        
    # CURRENT APPROXIMATE TRAIN TIME: 90 minutes
    def train_model(self):
        (x_train,y_train),(x_test,y_test) = self.data
        model = self.create_model()

        chk_point = ModelCheckpoint('best_model.tf',monitor='val_loss',verbose=1,save_best_only=True,mode='min') # Add patience?

        fit_model = model.fit(x_train,y_train.reshape(y_train.shape[0],y_train.shape[1],1),
            epochs=self.epochs,batch_size=self.batch_size,validation_split=.2,verbose=1,callbacks=[chk_point])
        return fit_model

    def performance_vis(self,fit_model):
        # Train-Validation Loss plot
        plt.figure()
        plt.plot(fit_model.history['loss'])
        plt.plot(fit_model.history['val_loss'])
        plt.legend(['Train','Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('training_visual.png')
        plt.show()

    def predict(self):
        best_model = load_model('best_model.tf')
        with open('data.pickle','rb') as data:
            sequence_data, test = cPickle.load(data)
        (x_train,y_train), (x_test,y_test) = sequence_data

        predictions = best_model.predict_classes(x_test.reshape((x_test.shape[0],x_test.shape[1])))

        with open('encoded_preds.pickle','wb') as encoded_preds:
            cPickle.dump(predictions,encoded_preds)        