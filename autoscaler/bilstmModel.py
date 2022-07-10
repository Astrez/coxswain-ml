from .abstractTrainer import Trainer
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional


class BiLSTMModel(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generateModel(self):
        self.model = Sequential()

        self.model.add(
            Bidirectional(
                LSTM(units = 64, return_sequences = True), 
                input_shape = (self.XTrain.shape[1], self.XTrain.shape[2])
            )
        )

        self.model.add(
            Bidirectional(LSTM(units = 64))
        )

        self.model.add(Dense(1))

        self.model.compile(optimizer='adam',loss='mse')

        print(self.model.summary())

        early_stop = keras.callbacks.EarlyStopping(
            monitor = 'val_loss', 
            patience = 10
        )

        self.model.fit(
            self.XTrain, 
            self.yTrain, 
            epochs = self.epochs, 
            validation_split = 0.2, 
            batch_size = self.batch, 
            shuffle = False, 
            callbacks = [early_stop]
        )

    

if __name__ == "__main__":
    lstm = BiLSTMModel()