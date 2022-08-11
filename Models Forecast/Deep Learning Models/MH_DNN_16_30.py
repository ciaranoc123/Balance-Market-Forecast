# BM Talos Multi-headed(MH) network year 1 hyperparameter search w/results
# full hyperparameter seaarch talos DNN/Multi-headed(MH) model

import os ;
# os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from datetime import timedelta as td
import traceback
# from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, LinearRegression
import numpy as np
import pandas as pd
from pandas import concat
import matplotlib.pyplot as plt
from functools import reduce
import importlib
import datetime as dt
from datetime import datetime
from math import floor
from bayes_opt import BayesianOptimization
import seaborn as sns
from math import sqrt

# tensorflow &keras
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras.layers import concatenate

from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
# from keras.layers import concatenate
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU

LeakyReLU = LeakyReLU(alpha=0.1)
from keras.activations import *

from keras.layers import LeakyReLU

LeakyReLU = LeakyReLU(alpha=0.1)

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn import preprocessing as prep
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, mean_absolute_error
from sklearn.metrics import mean_squared_error as MSE
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings

warnings.filterwarnings('ignore')

date_format = "%m/%d/%Y %H:%M"
date_parse = lambda date: dt.datetime.strptime(date, date_format)
# dat = pd.read_csv("C:/Users/ciara/OneDrive/Documents/BM_data.csv", index_col="SettlementPeriod", parse_dates=True, date_parser=date_parse)
# =============================================================================
# dat = pd.read_csv("/home/ciaran/Documents/BM_data.csv", index_col="SettlementPeriod", parse_dates=True, date_parser=date_parse)
dat = pd.read_csv("/home/coconnor/BM_data.csv", index_col="SettlementPeriod", parse_dates=True, date_parser=date_parse)
# =============================================================================

dat = dat.drop(["index"], axis=1)
dat = pd.DataFrame(dat)
dat = dat.bfill(axis='rows')
dat = dat.ffill(axis='rows')
dat = dat._get_numeric_data()

score_acc = make_scorer(mean_absolute_error)
mae = make_scorer(MAE, greater_is_better=False)
# i_shape=(None,X_train_Scaled.shape[1], X_train_Scaled.shape[2])

Y=dat.iloc[:, 0:16]
X=dat.iloc[:,16:]
X_train=X.iloc[16082:17186,:]
Y_train=Y.iloc[16082:17186,:]
X_test=X.iloc[17186:17523,:]
Y_test=Y.iloc[17186:17523,:]

rnn_train_LSTM_1 = X_train[
    ["lag_-3x1", "lag_-4x1", "lag_-5x1", "lag_-6x1", "lag_-7x1", "lag_-8x1", "lag_-9x1", "lag_-10x1", "lag_-11x1",
     "lag_-12x1", "lag_-13x1", "lag_-14x1", "lag_-15x1", "lag_-16x1", "lag_-17x1", "lag_-18x1", "lag_-19x1",
     "lag_-20x1", "lag_-21x1", "lag_-22x1", "lag_-23x1", "lag_-24x1", "lag_-25x1", "lag_-26x1", "lag_-27x1",
     "lag_-28x1", "lag_-29x1", "lag_-30x1", "lag_-31x1", "lag_-32x1", "lag_-33x1", "lag_-34x1", "lag_-35x1",
     "lag_-36x1", "lag_-37x1", "lag_-38x1", "lag_-39x1", "lag_-40x1", "lag_-41x1", "lag_-42x1", "lag_-43x1",
     "lag_-44x1", "lag_-45x1", "lag_-46x1", "lag_-47x1", "lag_-48x1", "lag_-49x1", "lag_-50x1"]]
rnn_train_LSTM_2 = X_train[
    ["lag_-3x2", "lag_-4x2", "lag_-5x2", "lag_-6x2", "lag_-7x2", "lag_-8x2", "lag_-9x2", "lag_-10x2", "lag_-11x2",
     "lag_-12x2", "lag_-13x2", "lag_-14x2", "lag_-15x2", "lag_-16x2", "lag_-17x2", "lag_-18x2", "lag_-19x2",
     "lag_-20x2", "lag_-21x2", "lag_-22x2", "lag_-23x2", "lag_-24x2", "lag_-25x2", "lag_-26x2", "lag_-27x2",
     "lag_-28x2", "lag_-29x2", "lag_-30x2", "lag_-31x2", "lag_-32x2", "lag_-33x2", "lag_-34x2", "lag_-35x2",
     "lag_-36x2", "lag_-37x2", "lag_-38x2", "lag_-39x2", "lag_-40x2", "lag_-41x2", "lag_-42x2", "lag_-43x2",
     "lag_-44x2", "lag_-45x2", "lag_-46x2", "lag_-47x2", "lag_-48x2", "lag_-49x2", "lag_-50x2"]]
rnn_train_LSTM_3 = X_train[
    ["lag_-2x3", "lag_-3x3", "lag_-4x3", "lag_-5x3", "lag_-6x3", "lag_-7x3", "lag_-8x3", "lag_-9x3", "lag_-10x3",
     "lag_-11x3", "lag_-12x3", "lag_-13x3", "lag_-14x3", "lag_-15x3", "lag_-16x3", "lag_-17x3", "lag_-18x3",
     "lag_-19x3", "lag_-20x3", "lag_-21x3", "lag_-22x3", "lag_-23x3", "lag_-24x3", "lag_-25x3", "lag_-26x3",
     "lag_-27x3", "lag_-28x3", "lag_-29x3", "lag_-30x3", "lag_-31x3", "lag_-32x3", "lag_-33x3", "lag_-34x3",
     "lag_-35x3", "lag_-36x3", "lag_-37x3", "lag_-38x3", "lag_-39x3", "lag_-40x3", "lag_-41x3", "lag_-42x3",
     "lag_-43x3", "lag_-44x3", "lag_-45x3", "lag_-46x3", "lag_-47x3", "lag_-48x3", "lag_-49x3"]]
rnn_train_LSTM_4 = X_train[
    ["lag_0x6", "lag_-1x6", "lag_-2x6", "lag_-3x6", "lag_-4x6", "lag_-5x6", "lag_-6x6", "lag_-7x6", "lag_-8x6",
     "lag_-9x6", "lag_-10x6", "lag_-11x6", "lag_-12x6", "lag_-13x6", "lag_-14x6", "lag_-15x6", "lag_-16x6", "lag_-17x6",
     "lag_-18x6", "lag_-19x6", "lag_-20x6", "lag_-21x6", "lag_-22x6", "lag_-23x6", "lag_-24x6", "lag_-25x6",
     "lag_-26x6", "lag_-27x6", "lag_-28x6", "lag_-29x6", "lag_-30x6", "lag_-31x6", "lag_-32x6", "lag_-33x6",
     "lag_-34x6", "lag_-35x6", "lag_-36x6", "lag_-37x6", "lag_-38x6", "lag_-39x6", "lag_-40x6", "lag_-41x6",
     "lag_-42x6", "lag_-43x6", "lag_-44x6", "lag_-45x6", "lag_-46x6", "lag_-47x6"]]
rnn_train_LSTM_5 = X_train[
    ["lag_-2x12", "lag_-3x12", "lag_-4x12", "lag_-5x12", "lag_-6x12", "lag_-7x12", "lag_-8x12", "lag_-9x12",
     "lag_-10x12", "lag_-11x12", "lag_-12x12", "lag_-13x12", "lag_-14x12", "lag_-15x12", "lag_-16x12", "lag_-17x12",
     "lag_-18x12", "lag_-19x12", "lag_-20x12", "lag_-21x12", "lag_-22x12", "lag_-23x12", "lag_-24x12", "lag_-25x12",
     "lag_-26x12", "lag_-27x12", "lag_-28x12", "lag_-29x12", "lag_-30x12", "lag_-31x12", "lag_-32x12", "lag_-33x12",
     "lag_-34x12", "lag_-35x12", "lag_-36x12", "lag_-37x12", "lag_-38x12", "lag_-39x12", "lag_-40x12", "lag_-41x12",
     "lag_-42x12", "lag_-43x12", "lag_-44x12", "lag_-45x12", "lag_-46x12", "lag_-47x12", "lag_-48x12", "lag_-49x12"]]

rnn_test_LSTM_1 = X_test[
    ["lag_-3x1", "lag_-4x1", "lag_-5x1", "lag_-6x1", "lag_-7x1", "lag_-8x1", "lag_-9x1", "lag_-10x1", "lag_-11x1",
     "lag_-12x1", "lag_-13x1", "lag_-14x1", "lag_-15x1", "lag_-16x1", "lag_-17x1", "lag_-18x1", "lag_-19x1",
     "lag_-20x1", "lag_-21x1", "lag_-22x1", "lag_-23x1", "lag_-24x1", "lag_-25x1", "lag_-26x1", "lag_-27x1",
     "lag_-28x1", "lag_-29x1", "lag_-30x1", "lag_-31x1", "lag_-32x1", "lag_-33x1", "lag_-34x1", "lag_-35x1",
     "lag_-36x1", "lag_-37x1", "lag_-38x1", "lag_-39x1", "lag_-40x1", "lag_-41x1", "lag_-42x1", "lag_-43x1",
     "lag_-44x1", "lag_-45x1", "lag_-46x1", "lag_-47x1", "lag_-48x1", "lag_-49x1", "lag_-50x1"]]
rnn_test_LSTM_2 = X_test[
    ["lag_-3x2", "lag_-4x2", "lag_-5x2", "lag_-6x2", "lag_-7x2", "lag_-8x2", "lag_-9x2", "lag_-10x2", "lag_-11x2",
     "lag_-12x2", "lag_-13x2", "lag_-14x2", "lag_-15x2", "lag_-16x2", "lag_-17x2", "lag_-18x2", "lag_-19x2",
     "lag_-20x2", "lag_-21x2", "lag_-22x2", "lag_-23x2", "lag_-24x2", "lag_-25x2", "lag_-26x2", "lag_-27x2",
     "lag_-28x2", "lag_-29x2", "lag_-30x2", "lag_-31x2", "lag_-32x2", "lag_-33x2", "lag_-34x2", "lag_-35x2",
     "lag_-36x2", "lag_-37x2", "lag_-38x2", "lag_-39x2", "lag_-40x2", "lag_-41x2", "lag_-42x2", "lag_-43x2",
     "lag_-44x2", "lag_-45x2", "lag_-46x2", "lag_-47x2", "lag_-48x2", "lag_-49x2", "lag_-50x2"]]
rnn_test_LSTM_3 = X_test[
    ["lag_-2x3", "lag_-3x3", "lag_-4x3", "lag_-5x3", "lag_-6x3", "lag_-7x3", "lag_-8x3", "lag_-9x3", "lag_-10x3",
     "lag_-11x3", "lag_-12x3", "lag_-13x3", "lag_-14x3", "lag_-15x3", "lag_-16x3", "lag_-17x3", "lag_-18x3",
     "lag_-19x3", "lag_-20x3", "lag_-21x3", "lag_-22x3", "lag_-23x3", "lag_-24x3", "lag_-25x3", "lag_-26x3",
     "lag_-27x3", "lag_-28x3", "lag_-29x3", "lag_-30x3", "lag_-31x3", "lag_-32x3", "lag_-33x3", "lag_-34x3",
     "lag_-35x3", "lag_-36x3", "lag_-37x3", "lag_-38x3", "lag_-39x3", "lag_-40x3", "lag_-41x3", "lag_-42x3",
     "lag_-43x3", "lag_-44x3", "lag_-45x3", "lag_-46x3", "lag_-47x3", "lag_-48x3", "lag_-49x3"]]
rnn_test_LSTM_4 = X_test[
    ["lag_0x6", "lag_-1x6", "lag_-2x6", "lag_-3x6", "lag_-4x6", "lag_-5x6", "lag_-6x6", "lag_-7x6", "lag_-8x6",
     "lag_-9x6", "lag_-10x6", "lag_-11x6", "lag_-12x6", "lag_-13x6", "lag_-14x6", "lag_-15x6", "lag_-16x6", "lag_-17x6",
     "lag_-18x6", "lag_-19x6", "lag_-20x6", "lag_-21x6", "lag_-22x6", "lag_-23x6", "lag_-24x6", "lag_-25x6",
     "lag_-26x6", "lag_-27x6", "lag_-28x6", "lag_-29x6", "lag_-30x6", "lag_-31x6", "lag_-32x6", "lag_-33x6",
     "lag_-34x6", "lag_-35x6", "lag_-36x6", "lag_-37x6", "lag_-38x6", "lag_-39x6", "lag_-40x6", "lag_-41x6",
     "lag_-42x6", "lag_-43x6", "lag_-44x6", "lag_-45x6", "lag_-46x6", "lag_-47x6"]]
rnn_test_LSTM_5 = X_test[
    ["lag_-2x12", "lag_-3x12", "lag_-4x12", "lag_-5x12", "lag_-6x12", "lag_-7x12", "lag_-8x12", "lag_-9x12",
     "lag_-10x12", "lag_-11x12", "lag_-12x12", "lag_-13x12", "lag_-14x12", "lag_-15x12", "lag_-16x12", "lag_-17x12",
     "lag_-18x12", "lag_-19x12", "lag_-20x12", "lag_-21x12", "lag_-22x12", "lag_-23x12", "lag_-24x12", "lag_-25x12",
     "lag_-26x12", "lag_-27x12", "lag_-28x12", "lag_-29x12", "lag_-30x12", "lag_-31x12", "lag_-32x12", "lag_-33x12",
     "lag_-34x12", "lag_-35x12", "lag_-36x12", "lag_-37x12", "lag_-38x12", "lag_-39x12", "lag_-40x12", "lag_-41x12",
     "lag_-42x12", "lag_-43x12", "lag_-44x12", "lag_-45x12", "lag_-46x12", "lag_-47x12", "lag_-48x12", "lag_-49x12"]]

rnn_train_ffnn_1 = X_train[
    ["lag_2x7", "lag_3x7", "lag_4x7", "lag_5x7", "lag_6x7", "lag_7x7", "lag_8x7", "lag_9x7", "lag_10x7", "lag_11x7",
     "lag_12x7", "lag_13x7", "lag_14x7", "lag_15x7", "lag_16x7", "lag_17x7"]]
rnn_train_ffnn_2 = X_train[
    ["lag_2x8", "lag_3x8", "lag_4x8", "lag_5x8", "lag_6x8", "lag_7x8", "lag_8x8", "lag_9x8", "lag_10x8", "lag_11x8",
     "lag_12x8", "lag_13x8", "lag_14x8", "lag_15x8", "lag_16x8", "lag_17x8"]]
rnn_train_ffnn_3 = X_train[
    ["lag_2x9", "lag_3x9", "lag_4x9", "lag_5x9", "lag_6x9", "lag_7x9", "lag_8x9", "lag_9x9", "lag_10x9", "lag_11x9",
     "lag_12x9", "lag_13x9", "lag_14x9", "lag_15x9", "lag_16x9", "lag_17x9"]]
rnn_train_ffnn_4 = X_train[
    ["lag_2x10", "lag_3x10", "lag_4x10", "lag_5x10", "lag_6x10", "lag_7x10", "lag_8x10", "lag_9x10", "lag_10x10",
     "lag_11x10", "lag_12x10", "lag_13x10", "lag_14x10", "lag_15x10", "lag_16x10", "lag_17x10"]]
rnn_train_ffnn_5 = X_train[
    ["lag_2x11", "lag_3x11", "lag_4x11", "lag_5x11", "lag_6x11", "lag_7x11", "lag_8x11", "lag_9x11", "lag_10x11",
     "lag_11x11", "lag_12x11", "lag_13x11", "lag_14x11", "lag_15x11", "lag_16x11", "lag_17x11"]]

rnn_test_ffnn_1 = X_test[
    ["lag_2x7", "lag_3x7", "lag_4x7", "lag_5x7", "lag_6x7", "lag_7x7", "lag_8x7", "lag_9x7", "lag_10x7", "lag_11x7",
     "lag_12x7", "lag_13x7", "lag_14x7", "lag_15x7", "lag_16x7", "lag_17x7"]]
rnn_test_ffnn_2 = X_test[
    ["lag_2x8", "lag_3x8", "lag_4x8", "lag_5x8", "lag_6x8", "lag_7x8", "lag_8x8", "lag_9x8", "lag_10x8", "lag_11x8",
     "lag_12x8", "lag_13x8", "lag_14x8", "lag_15x8", "lag_16x8", "lag_17x8"]]
rnn_test_ffnn_3 = X_test[
    ["lag_2x9", "lag_3x9", "lag_4x9", "lag_5x9", "lag_6x9", "lag_7x9", "lag_8x9", "lag_9x9", "lag_10x9", "lag_11x9",
     "lag_12x9", "lag_13x9", "lag_14x9", "lag_15x9", "lag_16x9", "lag_17x9"]]
rnn_test_ffnn_4 = X_test[
    ["lag_2x10", "lag_3x10", "lag_4x10", "lag_5x10", "lag_6x10", "lag_7x10", "lag_8x10", "lag_9x10", "lag_10x10",
     "lag_11x10", "lag_12x10", "lag_13x10", "lag_14x10", "lag_15x10", "lag_16x10", "lag_17x10"]]
rnn_test_ffnn_5 = X_test[
    ["lag_2x11", "lag_3x11", "lag_4x11", "lag_5x11", "lag_6x11", "lag_7x11", "lag_8x11", "lag_9x11", "lag_10x11",
     "lag_11x11", "lag_12x11", "lag_13x11", "lag_14x11", "lag_15x11", "lag_16x11", "lag_17x11"]]

rnn_Y = Y_train[
    ["lag_2y", "lag_3y", "lag_4y", "lag_5y", "lag_6y", "lag_7y", "lag_8y", "lag_9y", "lag_10y", "lag_11y", "lag_12y",
     "lag_13y", "lag_14y", "lag_15y", "lag_16y", "lag_17y"]]

X_scaler_LSTM_1 = preprocessing.MinMaxScaler()
X_scaler_LSTM_2 = preprocessing.MinMaxScaler()
X_scaler_LSTM_3 = preprocessing.MinMaxScaler()
X_scaler_LSTM_4 = preprocessing.MinMaxScaler()
X_scaler_LSTM_5 = preprocessing.MinMaxScaler()

X_scaler_ffnn_1 = preprocessing.MinMaxScaler()
X_scaler_ffnn_2 = preprocessing.MinMaxScaler()
X_scaler_ffnn_3 = preprocessing.MinMaxScaler()
X_scaler_ffnn_4 = preprocessing.MinMaxScaler()
X_scaler_ffnn_5 = preprocessing.MinMaxScaler()

Y_scaler = preprocessing.MinMaxScaler()

rnn_scaled_train_LSTM_1 = X_scaler_LSTM_1.fit_transform(rnn_train_LSTM_1)
rnn_scaled_train_LSTM_2 = X_scaler_LSTM_2.fit_transform(rnn_train_LSTM_2)
rnn_scaled_train_LSTM_3 = X_scaler_LSTM_3.fit_transform(rnn_train_LSTM_3)
rnn_scaled_train_LSTM_4 = X_scaler_LSTM_4.fit_transform(rnn_train_LSTM_4)
rnn_scaled_train_LSTM_5 = X_scaler_LSTM_5.fit_transform(rnn_train_LSTM_5)

rnn_scaled_train_ffnn_1 = X_scaler_ffnn_1.fit_transform(rnn_train_ffnn_1)
rnn_scaled_train_ffnn_2 = X_scaler_ffnn_2.fit_transform(rnn_train_ffnn_2)
rnn_scaled_train_ffnn_3 = X_scaler_ffnn_3.fit_transform(rnn_train_ffnn_3)
rnn_scaled_train_ffnn_4 = X_scaler_ffnn_4.fit_transform(rnn_train_ffnn_4)
rnn_scaled_train_ffnn_5 = X_scaler_ffnn_5.fit_transform(rnn_train_ffnn_5)

Y_train_Scaled = Y_scaler.fit_transform(Y_train)
Y_test_scaled = Y_scaler.transform(Y_test)

X_train_Scaled_LSTM = np.hstack(
    (rnn_scaled_train_LSTM_1, rnn_scaled_train_LSTM_2, rnn_scaled_train_LSTM_3,
     rnn_scaled_train_LSTM_4, rnn_scaled_train_LSTM_5)
).reshape(rnn_train_LSTM_1.shape[0], 5, 48).transpose(0, 2, 1)

X_train_Scaled_ffnn = np.hstack(
    (rnn_scaled_train_ffnn_1, rnn_scaled_train_ffnn_2, rnn_scaled_train_ffnn_3,
     rnn_scaled_train_ffnn_4, rnn_scaled_train_ffnn_5)
).reshape(rnn_train_ffnn_1.shape[0], 5, 16).transpose(0, 2, 1)

X_test_Scaled_LSTM = np.hstack(
    (X_scaler_LSTM_1.transform(rnn_test_LSTM_1), X_scaler_LSTM_2.transform(rnn_test_LSTM_2),
     X_scaler_LSTM_3.transform(rnn_test_LSTM_3), X_scaler_LSTM_4.transform(rnn_test_LSTM_4),
     X_scaler_LSTM_5.transform(rnn_test_LSTM_5))
).reshape(rnn_test_LSTM_1.shape[0], 5, 48).transpose(0, 2, 1)

X_test_Scaled_ffnn = np.hstack(
    (X_scaler_ffnn_1.transform(rnn_test_ffnn_1), X_scaler_ffnn_2.transform(rnn_test_ffnn_2),
     X_scaler_ffnn_3.transform(rnn_test_ffnn_3), X_scaler_ffnn_4.transform(rnn_test_ffnn_4),
     X_scaler_ffnn_5.transform(rnn_test_ffnn_5))
).reshape(rnn_test_ffnn_1.shape[0], 5, 16).transpose(0, 2, 1)

i_shape_lstm = (X_train_Scaled_LSTM.shape[1], X_train_Scaled_LSTM.shape[2])
i_shape_ffnn = (X_train_Scaled_ffnn.shape[1], X_train_Scaled_ffnn.shape[2])


def generate_train_and_test_dataframes(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                       test_start_time: dt, test_end_time: dt):
    """
    This method takes the raw information contained in the participat_df (i.e. explanatory variables and targets) and produces dataframes
        train_X, train_y, test_X, test_y, test_df
    What are the uses of these dataframes?
        - The train_X and train_y dataframes can be used to train models.
        - For the trained model, predictions can then be made using the test_X dataframe.
        - Predictions made in the previous step can then be compared to actual/target values contained in the test_y dataframe.
        - Finally, the test_df is used by other methods for plotting.
    Thus, this method will be called repeatedly in the rolling_walk_forward_validation method/process.

    Parameters
    ----------
    participant_df : pd.DataFrame
        Pandas dataframe, contains the participant time series info (i.e. explanatory and target variables, the index will be te trading period).
    date_time_column : str
        This is the column in the participant_df which indicates the deliveryperiod.
    train_start_time : dt
        The train_X and train_y dataframes will cover the interval [train_start_time, train_end_time].
    train_end_time : dt
        See previous comment.
    test_start_time : dt
        The test_X and test_y dataframes will cover the 24 trading periods from [train_end_time, train_end_time + 24 hours].
    test_end_time : dt
        See previous comment.
    columns_to_exclude: [str]
        These are the columns participant_df which should be ignored i.e. columns we don't want to use as explanatory variables.
    features_to_encode: [str]
        These are the categorical columns for which we want to apply one hot encoding.
    prefix_to_include: [str]
        For the categorical columns to which we apply one hot encoding, this list helps inform the naming convention for the newly created columns.
    targets: [str]
        These are the columns that we are trying to predict.

    Returns
    -------
    A tuple of dataframes.
                train_X, train_y, test_X, test_y,test_df
    Details and use cases for these dataframes are described above.
    """

    # These are the dataframes that will be returned from the method.
    train_X_LSTM = None
    train_X_ffnn = None

    train_y = None
    test_X_LSTM = None
    test_X_ffnn = None
    test_y = None
    test_df = None
    train_df = None

    try:

        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
        #             return train_X, train_y, test_X, test_y, test_df

        original_columns = list(participant_df.columns)

        # Remove any rows with nan's etc (there shouldn't be any in the input).
        participant_df = participant_df.dropna()

        date_format = "%m/%d/%Y %H:%M"

        # The train dataframe, it will be used later to create train_X and train_y.
        train_df = None
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
        train_df = participant_df[
            (participant_df.index >= train_start_time_str) & (participant_df.index < train_end_time_str)].copy(
            deep="True")

        if train_df is None or len(train_df) == 0:
            print(
                "Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")
        #             return train_X, train_y, test_X, test_y, test_df

        # Create the test dataframe, it will be used later to create test_X and test_y
        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format)
        test_df = participant_df[
            (participant_df.index >= test_start_time_str) & (participant_df.index < test_end_time_str)].copy(
            deep="True")

        if test_df is None or len(test_df) == 0:
            print(
                "Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")
        #             return train_X, train_y, test_X, test_y, test_df

        rnn_train_LSTM_1 = train_df[
            ["lag_-3x1", "lag_-4x1", "lag_-5x1", "lag_-6x1", "lag_-7x1", "lag_-8x1", "lag_-9x1", "lag_-10x1",
             "lag_-11x1",
             "lag_-12x1", "lag_-13x1", "lag_-14x1", "lag_-15x1", "lag_-16x1", "lag_-17x1", "lag_-18x1", "lag_-19x1",
             "lag_-20x1", "lag_-21x1", "lag_-22x1", "lag_-23x1", "lag_-24x1", "lag_-25x1", "lag_-26x1", "lag_-27x1",
             "lag_-28x1", "lag_-29x1", "lag_-30x1", "lag_-31x1", "lag_-32x1", "lag_-33x1", "lag_-34x1", "lag_-35x1",
             "lag_-36x1", "lag_-37x1", "lag_-38x1", "lag_-39x1", "lag_-40x1", "lag_-41x1", "lag_-42x1", "lag_-43x1",
             "lag_-44x1", "lag_-45x1", "lag_-46x1", "lag_-47x1", "lag_-48x1", "lag_-49x1", "lag_-50x1"]]
        rnn_train_LSTM_2 = train_df[
            ["lag_-3x2", "lag_-4x2", "lag_-5x2", "lag_-6x2", "lag_-7x2", "lag_-8x2", "lag_-9x2", "lag_-10x2",
             "lag_-11x2",
             "lag_-12x2", "lag_-13x2", "lag_-14x2", "lag_-15x2", "lag_-16x2", "lag_-17x2", "lag_-18x2", "lag_-19x2",
             "lag_-20x2", "lag_-21x2", "lag_-22x2", "lag_-23x2", "lag_-24x2", "lag_-25x2", "lag_-26x2", "lag_-27x2",
             "lag_-28x2", "lag_-29x2", "lag_-30x2", "lag_-31x2", "lag_-32x2", "lag_-33x2", "lag_-34x2", "lag_-35x2",
             "lag_-36x2", "lag_-37x2", "lag_-38x2", "lag_-39x2", "lag_-40x2", "lag_-41x2", "lag_-42x2", "lag_-43x2",
             "lag_-44x2", "lag_-45x2", "lag_-46x2", "lag_-47x2", "lag_-48x2", "lag_-49x2", "lag_-50x2"]]
        rnn_train_LSTM_3 = train_df[
            ["lag_-2x3", "lag_-3x3", "lag_-4x3", "lag_-5x3", "lag_-6x3", "lag_-7x3", "lag_-8x3", "lag_-9x3",
             "lag_-10x3",
             "lag_-11x3", "lag_-12x3", "lag_-13x3", "lag_-14x3", "lag_-15x3", "lag_-16x3", "lag_-17x3", "lag_-18x3",
             "lag_-19x3", "lag_-20x3", "lag_-21x3", "lag_-22x3", "lag_-23x3", "lag_-24x3", "lag_-25x3", "lag_-26x3",
             "lag_-27x3", "lag_-28x3", "lag_-29x3", "lag_-30x3", "lag_-31x3", "lag_-32x3", "lag_-33x3", "lag_-34x3",
             "lag_-35x3", "lag_-36x3", "lag_-37x3", "lag_-38x3", "lag_-39x3", "lag_-40x3", "lag_-41x3", "lag_-42x3",
             "lag_-43x3", "lag_-44x3", "lag_-45x3", "lag_-46x3", "lag_-47x3", "lag_-48x3", "lag_-49x3"]]
        rnn_train_LSTM_4 = train_df[
            ["lag_0x6", "lag_-1x6", "lag_-2x6", "lag_-3x6", "lag_-4x6", "lag_-5x6", "lag_-6x6", "lag_-7x6", "lag_-8x6",
             "lag_-9x6", "lag_-10x6", "lag_-11x6", "lag_-12x6", "lag_-13x6", "lag_-14x6", "lag_-15x6", "lag_-16x6",
             "lag_-17x6",
             "lag_-18x6", "lag_-19x6", "lag_-20x6", "lag_-21x6", "lag_-22x6", "lag_-23x6", "lag_-24x6", "lag_-25x6",
             "lag_-26x6", "lag_-27x6", "lag_-28x6", "lag_-29x6", "lag_-30x6", "lag_-31x6", "lag_-32x6", "lag_-33x6",
             "lag_-34x6", "lag_-35x6", "lag_-36x6", "lag_-37x6", "lag_-38x6", "lag_-39x6", "lag_-40x6", "lag_-41x6",
             "lag_-42x6", "lag_-43x6", "lag_-44x6", "lag_-45x6", "lag_-46x6", "lag_-47x6"]]
        rnn_train_LSTM_5 = train_df[
            ["lag_-2x12", "lag_-3x12", "lag_-4x12", "lag_-5x12", "lag_-6x12", "lag_-7x12", "lag_-8x12", "lag_-9x12",
             "lag_-10x12", "lag_-11x12", "lag_-12x12", "lag_-13x12", "lag_-14x12", "lag_-15x12", "lag_-16x12",
             "lag_-17x12",
             "lag_-18x12", "lag_-19x12", "lag_-20x12", "lag_-21x12", "lag_-22x12", "lag_-23x12", "lag_-24x12",
             "lag_-25x12",
             "lag_-26x12", "lag_-27x12", "lag_-28x12", "lag_-29x12", "lag_-30x12", "lag_-31x12", "lag_-32x12",
             "lag_-33x12",
             "lag_-34x12", "lag_-35x12", "lag_-36x12", "lag_-37x12", "lag_-38x12", "lag_-39x12", "lag_-40x12",
             "lag_-41x12",
             "lag_-42x12", "lag_-43x12", "lag_-44x12", "lag_-45x12", "lag_-46x12", "lag_-47x12", "lag_-48x12",
             "lag_-49x12"]]

        rnn_train_ffnn_1 = train_df[
            ["lag_2x7", "lag_3x7", "lag_4x7", "lag_5x7", "lag_6x7", "lag_7x7", "lag_8x7", "lag_9x7", "lag_10x7",
             "lag_11x7",
             "lag_12x7", "lag_13x7", "lag_14x7", "lag_15x7", "lag_16x7", "lag_17x7"]]
        rnn_train_ffnn_2 = train_df[
            ["lag_2x8", "lag_3x8", "lag_4x8", "lag_5x8", "lag_6x8", "lag_7x8", "lag_8x8", "lag_9x8", "lag_10x8",
             "lag_11x8",
             "lag_12x8", "lag_13x8", "lag_14x8", "lag_15x8", "lag_16x8", "lag_17x8"]]
        rnn_train_ffnn_3 = train_df[
            ["lag_2x9", "lag_3x9", "lag_4x9", "lag_5x9", "lag_6x9", "lag_7x9", "lag_8x9", "lag_9x9", "lag_10x9",
             "lag_11x9",
             "lag_12x9", "lag_13x9", "lag_14x9", "lag_15x9", "lag_16x9", "lag_17x9"]]
        rnn_train_ffnn_4 = train_df[
            ["lag_2x10", "lag_3x10", "lag_4x10", "lag_5x10", "lag_6x10", "lag_7x10", "lag_8x10", "lag_9x10",
             "lag_10x10",
             "lag_11x10", "lag_12x10", "lag_13x10", "lag_14x10", "lag_15x10", "lag_16x10", "lag_17x10"]]
        rnn_train_ffnn_5 = train_df[
            ["lag_2x11", "lag_3x11", "lag_4x11", "lag_5x11", "lag_6x11", "lag_7x11", "lag_8x11", "lag_9x11",
             "lag_10x11",
             "lag_11x11", "lag_12x11", "lag_13x11", "lag_14x11", "lag_15x11", "lag_16x11", "lag_17x11"]]

        rnn_test_LSTM_1 = test_df[
            ["lag_-3x1", "lag_-4x1", "lag_-5x1", "lag_-6x1", "lag_-7x1", "lag_-8x1", "lag_-9x1", "lag_-10x1",
             "lag_-11x1",
             "lag_-12x1", "lag_-13x1", "lag_-14x1", "lag_-15x1", "lag_-16x1", "lag_-17x1", "lag_-18x1", "lag_-19x1",
             "lag_-20x1", "lag_-21x1", "lag_-22x1", "lag_-23x1", "lag_-24x1", "lag_-25x1", "lag_-26x1", "lag_-27x1",
             "lag_-28x1", "lag_-29x1", "lag_-30x1", "lag_-31x1", "lag_-32x1", "lag_-33x1", "lag_-34x1", "lag_-35x1",
             "lag_-36x1", "lag_-37x1", "lag_-38x1", "lag_-39x1", "lag_-40x1", "lag_-41x1", "lag_-42x1", "lag_-43x1",
             "lag_-44x1", "lag_-45x1", "lag_-46x1", "lag_-47x1", "lag_-48x1", "lag_-49x1", "lag_-50x1"]]
        rnn_test_LSTM_2 = test_df[
            ["lag_-3x2", "lag_-4x2", "lag_-5x2", "lag_-6x2", "lag_-7x2", "lag_-8x2", "lag_-9x2", "lag_-10x2",
             "lag_-11x2",
             "lag_-12x2", "lag_-13x2", "lag_-14x2", "lag_-15x2", "lag_-16x2", "lag_-17x2", "lag_-18x2", "lag_-19x2",
             "lag_-20x2", "lag_-21x2", "lag_-22x2", "lag_-23x2", "lag_-24x2", "lag_-25x2", "lag_-26x2", "lag_-27x2",
             "lag_-28x2", "lag_-29x2", "lag_-30x2", "lag_-31x2", "lag_-32x2", "lag_-33x2", "lag_-34x2", "lag_-35x2",
             "lag_-36x2", "lag_-37x2", "lag_-38x2", "lag_-39x2", "lag_-40x2", "lag_-41x2", "lag_-42x2", "lag_-43x2",
             "lag_-44x2", "lag_-45x2", "lag_-46x2", "lag_-47x2", "lag_-48x2", "lag_-49x2", "lag_-50x2"]]
        rnn_test_LSTM_3 = test_df[
            ["lag_-2x3", "lag_-3x3", "lag_-4x3", "lag_-5x3", "lag_-6x3", "lag_-7x3", "lag_-8x3", "lag_-9x3",
             "lag_-10x3",
             "lag_-11x3", "lag_-12x3", "lag_-13x3", "lag_-14x3", "lag_-15x3", "lag_-16x3", "lag_-17x3", "lag_-18x3",
             "lag_-19x3", "lag_-20x3", "lag_-21x3", "lag_-22x3", "lag_-23x3", "lag_-24x3", "lag_-25x3", "lag_-26x3",
             "lag_-27x3", "lag_-28x3", "lag_-29x3", "lag_-30x3", "lag_-31x3", "lag_-32x3", "lag_-33x3", "lag_-34x3",
             "lag_-35x3", "lag_-36x3", "lag_-37x3", "lag_-38x3", "lag_-39x3", "lag_-40x3", "lag_-41x3", "lag_-42x3",
             "lag_-43x3", "lag_-44x3", "lag_-45x3", "lag_-46x3", "lag_-47x3", "lag_-48x3", "lag_-49x3"]]
        rnn_test_LSTM_4 = test_df[
            ["lag_0x6", "lag_-1x6", "lag_-2x6", "lag_-3x6", "lag_-4x6", "lag_-5x6", "lag_-6x6", "lag_-7x6", "lag_-8x6",
             "lag_-9x6", "lag_-10x6", "lag_-11x6", "lag_-12x6", "lag_-13x6", "lag_-14x6", "lag_-15x6", "lag_-16x6",
             "lag_-17x6",
             "lag_-18x6", "lag_-19x6", "lag_-20x6", "lag_-21x6", "lag_-22x6", "lag_-23x6", "lag_-24x6", "lag_-25x6",
             "lag_-26x6", "lag_-27x6", "lag_-28x6", "lag_-29x6", "lag_-30x6", "lag_-31x6", "lag_-32x6", "lag_-33x6",
             "lag_-34x6", "lag_-35x6", "lag_-36x6", "lag_-37x6", "lag_-38x6", "lag_-39x6", "lag_-40x6", "lag_-41x6",
             "lag_-42x6", "lag_-43x6", "lag_-44x6", "lag_-45x6", "lag_-46x6", "lag_-47x6"]]
        rnn_test_LSTM_5 = test_df[
            ["lag_-2x12", "lag_-3x12", "lag_-4x12", "lag_-5x12", "lag_-6x12", "lag_-7x12", "lag_-8x12", "lag_-9x12",
             "lag_-10x12", "lag_-11x12", "lag_-12x12", "lag_-13x12", "lag_-14x12", "lag_-15x12", "lag_-16x12",
             "lag_-17x12",
             "lag_-18x12", "lag_-19x12", "lag_-20x12", "lag_-21x12", "lag_-22x12", "lag_-23x12", "lag_-24x12",
             "lag_-25x12",
             "lag_-26x12", "lag_-27x12", "lag_-28x12", "lag_-29x12", "lag_-30x12", "lag_-31x12", "lag_-32x12",
             "lag_-33x12",
             "lag_-34x12", "lag_-35x12", "lag_-36x12", "lag_-37x12", "lag_-38x12", "lag_-39x12", "lag_-40x12",
             "lag_-41x12",
             "lag_-42x12", "lag_-43x12", "lag_-44x12", "lag_-45x12", "lag_-46x12", "lag_-47x12", "lag_-48x12",
             "lag_-49x12"]]

        rnn_test_ffnn_1 = test_df[
            ["lag_2x7", "lag_3x7", "lag_4x7", "lag_5x7", "lag_6x7", "lag_7x7", "lag_8x7", "lag_9x7", "lag_10x7",
             "lag_11x7",
             "lag_12x7", "lag_13x7", "lag_14x7", "lag_15x7", "lag_16x7", "lag_17x7"]]
        rnn_test_ffnn_2 = test_df[
            ["lag_2x8", "lag_3x8", "lag_4x8", "lag_5x8", "lag_6x8", "lag_7x8", "lag_8x8", "lag_9x8", "lag_10x8",
             "lag_11x8",
             "lag_12x8", "lag_13x8", "lag_14x8", "lag_15x8", "lag_16x8", "lag_17x8"]]
        rnn_test_ffnn_3 = test_df[
            ["lag_2x9", "lag_3x9", "lag_4x9", "lag_5x9", "lag_6x9", "lag_7x9", "lag_8x9", "lag_9x9", "lag_10x9",
             "lag_11x9",
             "lag_12x9", "lag_13x9", "lag_14x9", "lag_15x9", "lag_16x9", "lag_17x9"]]
        rnn_test_ffnn_4 = test_df[
            ["lag_2x10", "lag_3x10", "lag_4x10", "lag_5x10", "lag_6x10", "lag_7x10", "lag_8x10", "lag_9x10",
             "lag_10x10",
             "lag_11x10", "lag_12x10", "lag_13x10", "lag_14x10", "lag_15x10", "lag_16x10", "lag_17x10"]]
        rnn_test_ffnn_5 = test_df[
            ["lag_2x11", "lag_3x11", "lag_4x11", "lag_5x11", "lag_6x11", "lag_7x11", "lag_8x11", "lag_9x11",
             "lag_10x11",
             "lag_11x11", "lag_12x11", "lag_13x11", "lag_14x11", "lag_15x11", "lag_16x11", "lag_17x11"]]

        rnn_Y = train_df[
            ["lag_2y", "lag_3y", "lag_4y", "lag_5y", "lag_6y", "lag_7y", "lag_8y", "lag_9y", "lag_10y", "lag_11y",
             "lag_12y",
             "lag_13y", "lag_14y", "lag_15y", "lag_16y", "lag_17y"]]

        X_scaler_LSTM_1 = preprocessing.MinMaxScaler()
        X_scaler_LSTM_2 = preprocessing.MinMaxScaler()
        X_scaler_LSTM_3 = preprocessing.MinMaxScaler()
        X_scaler_LSTM_4 = preprocessing.MinMaxScaler()
        X_scaler_LSTM_5 = preprocessing.MinMaxScaler()

        X_scaler_ffnn_1 = preprocessing.MinMaxScaler()
        X_scaler_ffnn_2 = preprocessing.MinMaxScaler()
        X_scaler_ffnn_3 = preprocessing.MinMaxScaler()
        X_scaler_ffnn_4 = preprocessing.MinMaxScaler()
        X_scaler_ffnn_5 = preprocessing.MinMaxScaler()

        Y_scaler = preprocessing.MinMaxScaler()

        rnn_scaled_train_LSTM_1 = X_scaler_LSTM_1.fit_transform(rnn_train_LSTM_1)
        rnn_scaled_train_LSTM_2 = X_scaler_LSTM_2.fit_transform(rnn_train_LSTM_2)
        rnn_scaled_train_LSTM_3 = X_scaler_LSTM_3.fit_transform(rnn_train_LSTM_3)
        rnn_scaled_train_LSTM_4 = X_scaler_LSTM_4.fit_transform(rnn_train_LSTM_4)
        rnn_scaled_train_LSTM_5 = X_scaler_LSTM_5.fit_transform(rnn_train_LSTM_5)

        rnn_scaled_train_ffnn_1 = X_scaler_ffnn_1.fit_transform(rnn_train_ffnn_1)
        rnn_scaled_train_ffnn_2 = X_scaler_ffnn_2.fit_transform(rnn_train_ffnn_2)
        rnn_scaled_train_ffnn_3 = X_scaler_ffnn_3.fit_transform(rnn_train_ffnn_3)
        rnn_scaled_train_ffnn_4 = X_scaler_ffnn_4.fit_transform(rnn_train_ffnn_4)
        rnn_scaled_train_ffnn_5 = X_scaler_ffnn_5.fit_transform(rnn_train_ffnn_5)

        train_y = Y_scaler.fit_transform(rnn_Y)
        Y_scaler_n = Y_scaler.fit(rnn_Y)

        train_X_LSTM = np.hstack(
            (rnn_scaled_train_LSTM_1, rnn_scaled_train_LSTM_2, rnn_scaled_train_LSTM_3,
             rnn_scaled_train_LSTM_4, rnn_scaled_train_LSTM_5)
        ).reshape(rnn_train_LSTM_1.shape[0], 5, 48).transpose(0, 2, 1)

        train_X_ffnn = np.hstack(
            (rnn_scaled_train_ffnn_1, rnn_scaled_train_ffnn_2, rnn_scaled_train_ffnn_3,
             rnn_scaled_train_ffnn_4, rnn_scaled_train_ffnn_5)
        ).reshape(rnn_train_ffnn_1.shape[0], 5, 16).transpose(0, 2, 1)

        test_X_LSTM = np.hstack(
            (X_scaler_LSTM_1.transform(rnn_test_LSTM_1), X_scaler_LSTM_2.transform(rnn_test_LSTM_2),
             X_scaler_LSTM_3.transform(rnn_test_LSTM_3), X_scaler_LSTM_4.transform(rnn_test_LSTM_4),
             X_scaler_LSTM_5.transform(rnn_test_LSTM_5))
        ).reshape(rnn_test_LSTM_1.shape[0], 5, 48).transpose(0, 2, 1)

        test_X_ffnn = np.hstack(
            (X_scaler_ffnn_1.transform(rnn_test_ffnn_1), X_scaler_ffnn_2.transform(rnn_test_ffnn_2),
             X_scaler_ffnn_3.transform(rnn_test_ffnn_3), X_scaler_ffnn_4.transform(rnn_test_ffnn_4),
             X_scaler_ffnn_5.transform(rnn_test_ffnn_5))
        ).reshape(rnn_test_ffnn_1.shape[0], 5, 16).transpose(0, 2, 1)

        test_y = test_df.iloc[:, 0:16]

        return train_X_LSTM, train_X_ffnn, train_y, test_X_LSTM, test_X_ffnn, test_y, test_df, train_df, Y_scaler_n

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X_LSTM, train_X_ffnn, train_y, test_X_LSTM, test_X_ffnn, test_y, test_df, train_df, Y_scaler_n


def fit_multitarget_model(model, X_train_LSTM, X_train_ffnn, Y_train, X_test_LSTM, X_test_ffnn, Y_test,
                          actuals_and_forecast_df, targets, Y_scaler_n):
    """
    Fits the model to the training data.
    Then uses the test data to produce a forecast.
    Returns a dataframe containing the forecasts and actual values for each of the target variables.

    Parameters
    ----------
    model : object
        Model object i.e. randomforestregressor, linear model or other.
    X_train : dataframe
        The explanatory variables for the train/calibration set, numeric columns may already have been scaled.
    Y_train : dataframe
        The target variables for the train/calibration set. Might/Mighn't be scaled.
    X_test : dataframe
        The explanatory variables for the test set, columns may be scaled. It will comprise of 24 rows (1 row for each delivery period in the trading day).
    Y_test : dataframe
        The target variables for the test set i.e. what we would like to forecast. Similar to the previous bullet point, the dataframe will contain 24 rows.
    actuals_and_forecast_df : dataframe
        Initially the dataframe will only contain the actual values for each of the targets. At the end of the method it will also contain the forecast values.
    targets : [str]
        These are the items that we want to predict/forecast.
    scale_target_variables: boolean
       The target vector, do we want to scale it?

    Returns
    -------
    Returns a dataframe containing the forecasts and actual values for each of the target variables i.e. test set forecast and actuals.
    """
    try:
        Y_scaler = preprocessing.MinMaxScaler()
        Y_scaler = Y_scaler.fit(Y_train)
        cols = Y.columns.values.tolist()

        model.fit([X_train_LSTM, X_train_ffnn], Y_train)
        model_test_predictions = None
        #         model_test_predictions = model.predict(X_test)
        model_test_predictions = pd.DataFrame(
            Y_scaler_n.inverse_transform(model.predict([X_test_LSTM, X_test_ffnn]).reshape(1, 16)), columns=cols,
            index=Y_test.index)
        print(model_test_predictions)
        print("test number of observations: " + str(len(Y_test)))
        model_test_mse = mean_squared_error(Y_test, model_test_predictions)
        model_test_rmse = round(np.sqrt(model_test_mse), 2)
        model_test_mae = round(mean_absolute_error(Y_test, model_test_predictions), 2)
        print("test rmse: " + str(model_test_rmse))
        print("test mae: " + str(model_test_mae))

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast"] = model_test_predictions.iloc[:, i].tolist() if len(
                cols) > 1 else model_test_predictions.tolist()
            predictor_test_mse = mean_squared_error(Y_test[cols[i]], model_test_predictions.iloc[:, i]) if len(
                cols) > 1 else mean_squared_error(Y_test[cols[i]], model_test_predictions.tolist())
            predictor_test_rmse = round(np.sqrt(predictor_test_mse), 2)
            predictor_test_mae = round(mean_absolute_error(Y_test[cols[i]], model_test_predictions.iloc[:, i]),
                                       2) if len(cols) > 1 else round(
                mean_absolute_error(Y_test[cols[i]], model_test_predictions.tolist()), 2)
            print(cols[i] + " test rmse: " + str(predictor_test_rmse))
            print(cols[i] + " test mae: " + str(predictor_test_mae))

        Error_i = ([model_test_rmse, model_test_mae])
        print(Error_i)
        actuals_and_forecast_df = actuals_and_forecast_df.append(Error_i)

        # return the test set, the target and forecast values.
        #         actuals_and_forecast_df.sort_index(inplace=True)
        #         test_columns = ["SettlementPeriod"]
        #         for i in range(0,len(cols)):
        #             test_columns.extend([cols[i]+"_Forecast",cols[i]])
        #         actuals_and_forecast_df = actuals_and_forecast_df[test_columns]

        return actuals_and_forecast_df

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()


def rolling_walk_forward_validation(model, data, targets, start_time, end_time, training_days, path):
    """
    This method implements the rolling walk forward validation process.
    That is,
        (a) fit the model on the train data
        (b) use the fitted model on the test explanatory variables i.e. forecast 1 day ahead.
        (c) Move the training and test datasets forward by 1 day and repeat.
    The method will produce
        (1) A csv containing the forecast and actual target values i.e. test set output over the horizon of interest.
        (2) For each target variable, a graph of the actual and forecast values.

    Parameters
    ----------
    model: model
        The model that will be used to train the data and produce the 1 day ahead forecasts
    data: dataframe
        Participant dataframe containing the explanatory and target variables.
    explanatory_variables_of_interest : [str]
        The columns in data that will be used as explanatory variables when fitting the model.
        If there are categorical variables we want to use as explanatory variables, they are incorporated via the features_to_encode argument.
    targets: [str]
        The columns in data that we want to predict/forecast.
    features_to_encode: [str]
        If there are variables in data that we would like to apply one hot encoding to, we list them here. These one hot encoded vectors are then used as explanatory variables.
    prefix_to_include: [str]
        Just a string which will be used to name the columns if we apply one hot encoding (related to the features_to_encode argument).
     start_time: dt
       We will produce a forecast on unseen data for each trading period between [start_time, end_time].
     end_time: dt
       See previous point.
    training_days: int
       The number of training days (negative integer expected).
    path, unit_name, scenario: str
       The combination of path + unit_name + scenario indicate where the csv will be output to.
    scale_explanatory_variables: boolean
       The explanatory variables, do we want to scale them?
    scale_target_variables: boolean
       The target variables, do we want to scale them?

    Returns
    -------
    Output will include a  csv of the forecast/actual target values and a graph of the same.
    """
    try:

        all_columns = list(data.columns)
        results = pd.DataFrame()

        # Each time we
        # (a) fit the model on the calibration/train data
        # (b) apply it to the test data i.e. forecast 1 day ahead.
        # Repeat.
        date_format = "%m/%d/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)

        while start_time < end_time:

            # Train interval
            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time

            # Test interval, the test period is always the day ahead forecast
            test_start_time = train_end_time + td(hours=8)
            test_end_time = test_start_time + td(minutes=30)

            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))

            # Generate the calibration and test dataframes.
            train_X_LSTM, train_X_ffnn, train_y, test_X_LSTM, test_X_ffnn, test_y, test_df, train_df, Y_scaler_n = generate_train_and_test_dataframes(
                participant_df=dat, train_start_time=train_start_time.strftime("%m/%d/%Y %H:%M"),
                train_end_time=train_end_time.strftime("%m/%d/%Y %H:%M"),
                test_start_time=test_start_time.strftime("%m/%d/%Y %H:%M"),
                test_end_time=test_end_time.strftime("%m/%d/%Y %H:%M"))

            if train_X_LSTM is None or len(train_X_LSTM) == 0:
                print("Don't have a train dataframe for train_start_time: " + str(
                    train_start_time) + ", train_end_time: " + str(train_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue

            if test_X_LSTM is None or len(test_X_LSTM) == 0:
                print("Don't have a test dataframe for test_start_time: " + str(
                    test_start_time) + ", test_end_time: " + str(test_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue

            # Fit the model to the train datasets, produce a forecast and return a dataframe containing the forecast/actuals.
            actuals_and_forecast_df = fit_multitarget_model(model=model, Y_scaler_n=Y_scaler_n,
                                                            X_train_LSTM=train_X_LSTM, X_train_ffnn=train_X_ffnn,
                                                            Y_train=train_y,
                                                            X_test_LSTM=test_X_LSTM, X_test_ffnn=test_X_ffnn,
                                                            Y_test=test_y,
                                                            actuals_and_forecast_df=test_df.iloc[:, 0:16],
                                                            targets=Y.columns.values.tolist())

            results = results.append(actuals_and_forecast_df)

            start_time = start_time + td(hours=8)

        results.to_csv(path + ".csv", index=False)

    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()


def create_model():    
            visible1 = Input(shape=(i_shape_lstm))            
            dense1 = LSTM(16, return_sequences= True, activation='tanh' , input_shape=i_shape_lstm)(visible1)
            dense2 = LSTM(16, return_sequences= True, activation='tanh' , input_shape=i_shape_lstm)(dense1)
            do_lstm = Dropout(0.044444, seed=123)(dense2)
            dense3 = LSTM(16, return_sequences= True, activation='tanh' , input_shape=i_shape_lstm)(do_lstm)
            flat1 = Flatten()(dense3)    
    
            visible2 = Input(shape=(i_shape_ffnn))        
            dense5 = Dense(256, activation='sigmoid')(visible2)        
            dense6 = Dense(256, activation='sigmoid')(dense5)
            do_ffnn = Dropout(0.088889, seed=123)(dense6)
            dense7 = Dense(256, activation='sigmoid')(do_ffnn)
            flat2 = Flatten()(dense7)
    
            merged = concatenate([flat1, flat2])
            dense_f = Dense(256, activation=LeakyReLU)(merged)        
            outputs = Dense(16)(dense_f)

            model = Model(inputs=[visible1, visible2], outputs=outputs)    
            opt = Adam(lr=0.013367)
            model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
            return model
        
es = EarlyStopping(monitor='mean_absolute_error', mode='min', verbose=0, patience=20)
mmo = KerasRegressor(build_fn=create_model, epochs=300, batch_size=32, verbose=2, callbacks=[es])


rolling_walk_forward_validation(model=mmo, data=dat, start_time='12/1/2020 00:00', end_time='3/1/2021  00:00',
                                targets=Y.columns.values.tolist(), training_days=-30,
                                path="/home/coconnor/BM_results_MH_16_30")