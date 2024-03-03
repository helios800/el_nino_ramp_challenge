import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.base import BaseEstimator

class LSTMEstimator(BaseEstimator):
    def __init__(self, units=100, optimizer='adam', loss='mean_squared_error', epochs=30, batch_size=50):
        self.units = units
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        self.model = Sequential()
        self.model.add(LSTM(units=self.units, input_shape=(X.shape[1], 1)))
        self.model.add(Dense(units=1))
        
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)
        
        return self

    def predict(self, X):
        return np.reshape(self.model.predict(X),np.shape(self.model.predict(X))[0])
    
def get_estimator():
    return LSTMEstimator()
