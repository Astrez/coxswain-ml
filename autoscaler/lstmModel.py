from .abstractTrainer import Trainer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


class LSTMModel(Trainer):

    def __init__(self, *args, **kwargs):
        print(kwargs)
        super().__init__(*args, **kwargs)

    def generateModel(self):
        self.model=Sequential()
        self.model.add(
            LSTM(50, return_sequences = True, input_shape = (self.timeStep, 1))
        )

        self.model.add(
            LSTM(50, return_sequences = True)
        )

        self.model.add(LSTM(50))

        self.model.add(Dense(1))

        self.model.compile(loss='mean_squared_error', optimizer='adam')

        print(self.model.summary())

        self.model.fit(
            self.XTrain, 
            self.yTrain, 
            epochs = self.epochs, 
            batch_size = self.batch, 
            verbose = 1
        )

    

if __name__ == "__main__":
    lstm = LSTMModel()