from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler       
from random import randint
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os
import math


class Trainer(ABC):

    def __init__(self, *args, **kwargs):
        self.model = None
        self.dataScaler = MinMaxScaler(feature_range = (0, 1))
        print(kwargs)

        self.csv = kwargs.get('csv', None)
        self.algorithm = kwargs.get('algorithm', "LSTM")
        self.trainFraction = kwargs.get('trainFraction', 0.9)
        self.timeStep = kwargs.get('timeStep', 100)
        self.epochs = kwargs.get('epochs', 100)
        self.batch = kwargs.get('batchSize', 64)

        

        if self.csv:
            self.train()
        else:
            # Load weights
            self.loadWeights()
    
    @abstractmethod
    def generateModel(self):
        pass

    def train(self):
        self.preProcessing()
        self.dataTransform()
        self.generateModel()
        self.saveWeights()
        print("Here")
        # self.loadWeights()
        print("Here")
        self.predictionForTraining()
        self.loadMetrics()

        # Temporary code
        # testdata = [randint(1000, 9999) for _ in range(100)]
        # print("Testdata : \n", testdata)
        # print(f"Number of requested expected : {self.predict(testdata)}")
    
    def preProcessing(self):
        
        trafficData = pd.read_csv(self.csv)
        # Time format : 1998-04-30 21:30:00
        # trafficData['period'] = pd.to_datetime(trafficData.period, format='%Y-%m-%d %H:%M:%S')
        trafficData['period'] = pd.to_datetime(trafficData.Date, format='%m/%d/%Y')
        trafficData['Page.Loads'] = trafficData['Page.Loads'].str.replace(',', '')
        trafficData['count'] = trafficData['Page.Loads'].astype(str).astype(int)
        trafficData = trafficData[['period', 'count']]
        trafficData.set_index('period', inplace=True)
        trafficData = trafficData[trafficData.index <= '12/31/2019']

        self.data = trafficData.reset_index()['count']
    
    def dataTransform(self):
        self.data = self.dataScaler.fit_transform(np.array(self.data ).reshape(-1,1))
        trainingSize = int(len(self.data) * 0.9)

        self.trainData = self.data[0:trainingSize,:]
        self.testData = self.data[trainingSize:len(self.data),:1]

        self.XTrain, self.yTrain = self.createDataset(self.trainData)
        self.XTest, self.yTest = self.createDataset(self.testData)

        self.XTrain = self.XTrain.reshape(self.XTrain.shape[0], self.XTrain.shape[1] , 1)
        self.XTest = self.XTest.reshape(self.XTest.shape[0], self.XTest.shape[1] , 1)
        

    def createDataset(self, data):
        timeSeriesData = []
        toBePredicted = []
        for i in range(len(data) - self.timeStep-1):
            a = data[i:(i+self.timeStep), 0]
            timeSeriesData.append(a)
            toBePredicted.append([data[i + self.timeStep, 0]])
        return np.array(timeSeriesData), np.array(toBePredicted)

    def predictionForTraining(self):
        self.yTrain = self.dataScaler.inverse_transform(self.yTrain)
        self.yTest = self.dataScaler.inverse_transform(self.yTest)

        self.trainPredict = self.dataScaler.inverse_transform(self.model.predict(self.XTrain))
        self.testPredict = self.dataScaler.inverse_transform(self.model.predict(self.XTest))


    def predict(self, timeSeriesData : list):
        timeSeriesData = self.dataScaler.fit_transform(np.array(timeSeriesData).reshape(-1, 1))
        timeSeriesData = np.array([timeSeriesData])
        formatted = timeSeriesData.reshape(timeSeriesData.shape[0], timeSeriesData.shape[1], 1)
        value = self.dataScaler.inverse_transform(self.model.predict(formatted))
        return round(np.ndarray.tolist(value)[0][0])

    def saveWeights(self):
        self.model.save(os.path.join(os.getcwd(), "autoscaler", "models", f"{self.algorithm}.h5"))

    def loadWeights(self):
        self.model  = tf.keras.models.load_model(os.path.join(os.getcwd(), "autoscaler", "models", f"{self.algorithm}.h5"))

    def loadMetrics(self):
        mse = mean_squared_error(self.yTest, self.testPredict)
        rmse = math.sqrt(mse)
        # nrmse = rmse / (max(self.data['count']) - min(self.data['count']))
        # print("mse=",mse,"\t\trmse=",rmse,"\tnrmse=",nrmse)
        print(f'Mean Squared Error : {mse}')
        print(f'Root Mean Square Error : {rmse}')
        # print(f'Normalized Root Mean Square Error : {nrmse}')
        print(f'Root Mean Squared Log Error : {math.log(rmse)}')
        print(f'r2 Score : {r2_score(self.yTest, self.testPredict)}')
