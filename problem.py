import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from IPython.display import Image
import os 

from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import rampwf as rw
from sklearn.model_selection import ShuffleSplit


warnings.filterwarnings("ignore")



problem_title = 'El Nino time serie regression'
_target_column_name = 'standardized anomaly'

#define workflow for submitting the ramp test

Predictions = rw.prediction_types.make_regression()
workflow = rw.workflows.Estimator()

#RMSE as error type
score_types = [
    rw.score_types.RMSE(),
    rw.score_types.RelativeRMSE(name='rel_rmse'),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=1, test_size=0.1)
    return cv.split(X)




def _read_data(path, file_name):
    full_path = os.path.join(path, file_name)
    data = np.loadtxt(full_path)
    
    return data

def get_train_data(path='.'):
    f_name_x = 'x_train.txt'
    f_name_y = 'y_train.txt'

    return _read_data(path, f_name_x),_read_data(path, f_name_y)


def get_test_data(path='.'):
    f_name_x = 'x_test.txt'
    f_name_y = 'y_test.txt'
    return _read_data(path, f_name_x),_read_data(path, f_name_y)
