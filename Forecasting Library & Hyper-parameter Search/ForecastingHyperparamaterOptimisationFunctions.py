import os ;
# os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import importlib
from pandas import concat
from datetime import datetime
from datetime import timedelta as td
from functools import reduce
from math import floor, sqrt
from hyperopt import hp, fmin, tpe

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, metrics
from sklearn import
from sklearn import preprocessing as prep
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, mean_absolute_error
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_array
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression
from epftoolbox.models import evaluate_lear_in_test_dataset
from sklearn.linear_model import LassoLarsIC, Lasso
from epftoolbox.data import scaling, read_data
from epftoolbox.models import LEAR, evaluate_lear_in_test_dataset
from epftoolbox.evaluation import MAE, sMAPE
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.layers import concatenate, Flatten, Dense, BatchNormalization, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.layers import
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, GRU, Dense, LSTM, concatenate
from tensorflow.keras import optimizers, initializers
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.activations import *
import tensorflow.keras.backend as K
warnings.filterwarnings('ignore')

def generate_train_and_test_dataframes_SVR_XGB_RF(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                       test_start_time: dt, test_end_time: dt):
    train_X = None
    train_y = None
    test_X = None
    test_y = None
    test_df = None

    try:

        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
            return train_X, train_y, test_X, test_y, test_df

        original_columns = list(participant_df.columns)

        participant_df = participant_df.dropna()

        date_format = "%m/%d/%Y %H:%M"
        train_df = None

        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
        train_df = participant_df[
            (participant_df.index >= train_start_time_str) & (participant_df.index < train_end_time_str)].copy(
            deep="True")

        if train_df is None or len(train_df) == 0:
            print(
                "Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")
            return train_X, train_y, test_X, test_y, test_df

        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format)
        test_df = participant_df[
            (participant_df.index >= test_start_time_str) & (participant_df.index < test_end_time_str)].copy(
            deep="True")

        if test_df is None or len(test_df) == 0:
            print(
                "Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")
            return train_X, train_y, test_X, test_y, test_df

        train_X = train_df.iloc[:, 16:]
        test_X = test_df.iloc[:, 16:]
        train_y = train_df.iloc[:, 0:16]
        test_y = test_df.iloc[:, 0:16]

        return train_X, train_y, test_X, test_y, test_df

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df


def fit_multitarget_model_SVR_XGB_RF(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets):
    try:
        model.fit(X_train, Y_train)

        model_test_predictions = None
        model_train_predictions = None
        model_train_predictions = model.predict(X_train)
        model_test_predictions = model.predict(X_test)

        cols = Y_train.columns.values.tolist()

        model_train_mse = mean_squared_error(Y_train, model_train_predictions)
        model_train_rmse = round(np.sqrt(model_train_mse), 2)
        model_train_mae = round(mean_absolute_error(Y_train, model_train_predictions), 2)

        model_test_mse = mean_squared_error(Y_test, model_test_predictions)
        model_test_rmse = round(np.sqrt(model_test_mse), 2)
        model_test_mae = round(mean_absolute_error(Y_test, model_test_predictions), 2)

        for i in range(0, len(cols)):
            predictor_train_mse = mean_squared_error(Y_train[cols[i]], model_train_predictions[:, i]) if len(
                cols) > 1 else mean_squared_error(Y_train[cols[i]], model_train_predictions.tolist())
            predictor_train_rmse = round(np.sqrt(predictor_train_mse), 2)
            predictor_train_mae = round(mean_absolute_error(Y_train[cols[i]], model_train_predictions[:, i]), 2) if len(
                cols) > 1 else round(mean_absolute_error(Y_train[cols[i]], model_train_predictions.tolist()), 2)

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast"] = model_test_predictions[:, i].tolist() if len(
                cols) > 1 else model_test_predictions.tolist()
            predictor_test_mse = mean_squared_error(Y_test[cols[i]], model_test_predictions[:, i]) if len(
                cols) > 1 else mean_squared_error(Y_test[cols[i]], model_test_predictions.tolist())
            predictor_test_rmse = round(np.sqrt(predictor_test_mse), 2)
            predictor_test_mae = round(mean_absolute_error(Y_test[cols[i]], model_test_predictions[:, i]), 2) if len(
                cols) > 1 else round(mean_absolute_error(Y_test[cols[i]], model_test_predictions.tolist()), 2)

        Error_i = ([model_test_rmse, model_test_mae, model_train_rmse, model_train_mae])
        actuals_and_forecast_df = actuals_and_forecast_df.append(Error_i)

        return actuals_and_forecast_df

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()


def rolling_walk_forward_validation_SVR_XGB_RF(model, data, targets, start_time, end_time, training_days, path):
    try:

        all_columns = list(data.columns)
        results = pd.DataFrame()

        date_format = "%m/%d/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)

        while start_time < end_time:

            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time

            test_start_time = train_end_time + td(hours=8)
            test_end_time = test_start_time + td(minutes=30)

            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))

            train_X, train_y, test_X, test_y, test_df = generate_train_and_test_dataframes(participant_df=dat,
                                                                                           train_start_time=train_start_time.strftime(
                                                                                               "%m/%d/%Y %H:%M"),
                                                                                           train_end_time=train_end_time.strftime(
                                                                                               "%m/%d/%Y %H:%M"),
                                                                                           test_start_time=test_start_time.strftime(
                                                                                               "%m/%d/%Y %H:%M"),
                                                                                           test_end_time=test_end_time.strftime(
                                                                                               "%m/%d/%Y %H:%M"))

            if train_X is None or len(train_X) == 0:
                print("Don't have a train dataframe for train_start_time: " + str(
                    train_start_time) + ", train_end_time: " + str(train_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue

            if test_X is None or len(test_X) == 0:
                print("Don't have a test dataframe for test_start_time: " + str(
                    test_start_time) + ", test_end_time: " + str(test_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue

            actuals_and_forecast_df = fit_multitarget_model(model=model, X_train=train_X, Y_train=train_y,
                                                            X_test=test_X, Y_test=test_y,
                                                            actuals_and_forecast_df=test_df,
                                                            targets=Y.columns.values.tolist())

            results = results.append(actuals_and_forecast_df)
            start_time = start_time + td(hours=8)

        results.to_csv(path + ".csv", index=False)


    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()

class ARIMAModel(BaseEstimator):
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def fit(self, X, y):
        self.model = ARIMA(endog=y, order=self.order, seasonal_order=self.seasonal_order)
        self.model = self.model.fit()
        return self

    def predict(self, X):
        return self.model.forecast(steps=len(X))

    def get_params(self, deep=True):
        return {'order': self.order, 'seasonal_order': self.seasonal_order}

    def set_params(self, **params):
        self.order = params['order']
        self.seasonal_order = params['seasonal_order']
        return self


# Function to fit ARIMA model
def fit_arima_model(X, y):
    arima_models = []
    for i in range(y.shape[1]):
        param_grid = {
            'order': [(1, 1, 1), (2, 1, 1), (1, 1, 2)],  # Example parameter grid, adjust as needed
            'seasonal_order': [(0, 0, 0, 0)]  # Example seasonal parameter grid, adjust as needed
        }

        arima = ARIMAModel()
        grid_search = GridSearchCV(arima, param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X, y.iloc[:, i])

        best_params = grid_search.best_params_
        model = ARIMAModel(order=best_params['order'], seasonal_order=best_params['seasonal_order'])
        model.fit(X, y.iloc[:, i])
        arima_models.append(model)
    return arima_models

def generate_train_and_test_dataframes_LEAR(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                       test_start_time: dt, test_end_time: dt):


    train_X = None
    train_Y = None
    test_X = None
    test_Y = None
    test_df = None
    train_df = None

    try:

        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
        #             return train_X, train_y, test_X, test_y, test_df

        original_columns = list(participant_df.columns)

        participant_df = participant_df.dropna()

        date_format = "%m/%d/%Y %H:%M"

        train_df = None
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
        train_df = participant_df[
            (participant_df.index >= train_start_time_str) & (participant_df.index < train_end_time_str)].copy(
            deep="True")

        if train_df is None or len(train_df) == 0:
            print(
                "Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")

        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format)
        test_df = participant_df[
            (participant_df.index >= test_start_time_str) & (participant_df.index < test_end_time_str)].copy(
            deep="True")

        if test_df is None or len(test_df) == 0:
            print(
                "Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")

        X_1 = train_df.loc[:,"lag_-3x1":"lag_-50x1"]
        X_2 = train_df.loc[:,"lag_-3x2":"lag_-50x2"]
        X_3 = train_df.loc["lag_-2x3":"lag_-49x3"]
        X_4 = train_df.loc[:,"lag_0x6":"lag_-47x6"]
        X_5 = train_df.loc[:,"lag_-2x12":"lag_-49x12"]
        X_6 = train_df.loc[:,"lag_2x7":"lag_17x7"]
        X_7 = train_df.loc[:,"lag_2x8":"lag_17x8"]
        X_8 = train_df.loc[:,"lag_2x9":"lag_17x9"]
        X_9 = train_df.loc[:,"lag_2x10":"lag_17x10"]
        X_10 = train_df.loc[:,"lag_2x11": "lag_17x11"]

        X_test1 = test_df.loc[:,"lag_-3x1":"lag_-50x1"]
        X_test2 = test_df.loc[:,"lag_-3x2":"lag_-50x2"]
        X_test3 = test_df.loc[:,"lag_-2x3":"lag_-49x3"]
        X_test4 = test_df.loc[:,"lag_0x6":"lag_-47x6"]
        X_test5 = test_df.loc[:,"lag_-2x12":"lag_-49x12"]
        X_test6 = test_df.loc[:,"lag_2x7":"lag_17x7"]
        X_test7 = test_df.loc[:,"lag_2x8":"lag_17x8"]
        X_test8 = test_df.loc[:,"lag_2x9":"lag_17x9"]
        X_test9 = test_df.loc[:,"lag_2x10":"lag_17x10"]
        X_test10 = test_df.loc[:,"lag_2x11":"lag_17x11"]
        Y_1 = train_df.loc[:,"lag_2y": "lag_17y"]

        [X_1], X_scaler1 = scaling([X_1.values], 'Invariant')
        [X_2], X_scaler2 = scaling([X_2.values], 'Invariant')
        [X_3], X_scaler3 = scaling([X_3.values], 'Invariant')
        [X_4], X_scaler4 = scaling([X_4.values], 'Invariant')
        [X_5], X_scaler5 = scaling([X_5.values], 'Invariant')

        [X_6], X_scaler6 = scaling([X_6.values], 'Invariant')
        [X_7], X_scaler7 = scaling([X_7.values], 'Invariant')
        [X_8], X_scaler8 = scaling([X_8.values], 'Invariant')
        [X_9], X_scaler9 = scaling([X_9.values], 'Invariant')
        [X_10], X_scaler10 = scaling([X_10.values], 'Invariant')

        X_test_1= X_scaler1.transform(X_test1.values)
        X_test_2= X_scaler2.transform(X_test2.values)
        X_test_3= X_scaler3.transform(X_test3.values)
        X_test_4= X_scaler4.transform(X_test4.values)
        X_test_5= X_scaler5.transform(X_test5.values)
        X_test_6= X_scaler6.transform(X_test6.values)
        X_test_7= X_scaler7.transform(X_test7.values)
        X_test_8= X_scaler8.transform(X_test8.values)
        X_test_9= X_scaler9.transform(X_test9.values)
        X_test_10= X_scaler10.transform(X_test10.values)

        [train_Y], Y_scaler = scaling([Y_1.values], 'Invariant')
        Y_scaler_n = Y_scaler

        train_X=np.concatenate((X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10), axis=1)
        test_X=np.concatenate((X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6, X_test_7, X_test_8, X_test_9, X_test_10), axis=1)
        test_Y = test_df.iloc[:, 0:16]

        return train_X, train_Y, test_X, test_Y, test_df, train_df, Y_scaler_n

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_Y, test_X, test_Y, test_df, train_df, Y_scaler_n


def generate_train_and_test_dataframes_SH_DNN(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                       test_start_time: dt, test_end_time: dt):
    train_X = None
    train_y = None
    test_X = None
    test_y = None
    test_df = None
    train_df = None

    try:

        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")

        original_columns = list(participant_df.columns)

        participant_df = participant_df.dropna()

        date_format = "%m/%d/%Y %H:%M"

        train_df = None
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
        train_df = participant_df[
            (participant_df.index >= train_start_time_str) & (participant_df.index < train_end_time_str)].copy(
            deep="True")

        if train_df is None or len(train_df) == 0:
            print(
                "Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")

        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format)
        test_df = participant_df[
            (participant_df.index >= test_start_time_str) & (participant_df.index < test_end_time_str)].copy(
            deep="True")

        if test_df is None or len(test_df) == 0:
            print(
                "Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")

        rnn_train1_a = train_df.loc[:, "lag_-3x1":"lag_-18x1"]
        rnn_train1_b = train_df.loc[:, "lag_-19x1":"lag_-34x1"]
        rnn_train1_c = train_df.loc[:, "lag_-35x1":"lag_-50x1"]

        rnn_train2_a = train_df.loc[:, "lag_-3x2":"lag_-18x2"]
        rnn_train2_b = train_df.loc[:, "lag_-19x2":"lag_-34x2"]
        rnn_train2_c = train_df.loc[:, "lag_-35x2":"lag_-50x2"]

        rnn_train3_a = train_df.loc[:, "lag_-2x3":"lag_-17x3"]
        rnn_train3_b = train_df.loc[:, "lag_-18x3":"lag_-33x3"]
        rnn_train3_c = train_df.loc[:, "lag_-34x3":"lag_-49x3"]

        rnn_train4_a = train_df.loc[:, "lag_0x6":"lag_-15x6"]
        rnn_train4_b = train_df.loc[:, "lag_-16x6":"lag_-31x6"]
        rnn_train4_c = train_df.loc[:, "lag_-32x6":"lag_-47x6"]

        rnn_train5_a = train_df.loc[:, "lag_-2x12":"lag_-17x12"]
        rnn_train5_b = train_df.loc[:, "lag_-18x12":"lag_-33x12"]
        rnn_train5_c = train_df.loc[:, "lag_-34x12":"lag_-49x12"]

        rnn_train6 = train_df.loc[:, "lag_2x7":"lag_17x7"]
        rnn_train7 = train_df.loc[:, "lag_2x8":"lag_17x8"]
        rnn_train8 = train_df.loc[:, "lag_2x9":"lag_17x9"]
        rnn_train9 = train_df.loc[:, "lag_2x10":"lag_17x10"]
        rnn_train10 = train_df.loc[:, "lag_2x11":"lag_17x11"]

        rnn_test1_a = test_df.loc[:, "lag_-3x1":"lag_-18x1"]
        rnn_test1_b = test_df.loc[:, "lag_-19x1":"lag_-34x1"]
        rnn_test1_c = test_df.loc[:, "lag_-35x1":"lag_-50x1"]

        rnn_test2_a = test_df.loc[:, "lag_-3x2":"lag_-18x2"]
        rnn_test2_b = test_df.loc[:, "lag_-19x2":"lag_-34x2"]
        rnn_test2_c = test_df.loc[:, "lag_-35x2":"lag_-50x2"]

        rnn_test3_a = test_df.loc[:, "lag_-2x3":"lag_-17x3"]
        rnn_test3_b = test_df.loc[:, "lag_-18x3":"lag_-33x3"]
        rnn_test3_c = test_df.loc[:, "lag_-34x3":"lag_-49x3"]

        rnn_test4_a = test_df.loc[:, "lag_0x6":"lag_-15x6"]
        rnn_test4_b = test_df.loc[:, "lag_-16x6":"lag_-31x6"]
        rnn_test4_c = test_df.loc[:, "lag_-32x6":"lag_-47x6"]

        rnn_test5_a = test_df.loc[:, "lag_-2x12":"lag_-17x12"]
        rnn_test5_b = test_df.loc[:, "lag_-18x12":"lag_-33x12"]
        rnn_test5_c = test_df.loc[:, "lag_-34x12":"lag_-49x12"]

        rnn_test6 = test_df.loc[:, "lag_2x7":"lag_17x7"]
        rnn_test7 = test_df.loc[:, "lag_2x8":"lag_17x8"]
        rnn_test8 = test_df.loc[:, "lag_2x9":"lag_17x9"]
        rnn_test9 = test_df.loc[:, "lag_2x10":"lag_17x10"]
        rnn_test10 = test_df.loc[:, "lag_2x11":"lag_17x11"]

        rnn_Y = train_df.loc[:, "lag_2y": "lag_17y"]

        X_scaler1_a = preprocessing.MinMaxScaler()
        X_scaler1_b = preprocessing.MinMaxScaler()
        X_scaler1_c = preprocessing.MinMaxScaler()

        X_scaler2_a = preprocessing.MinMaxScaler()
        X_scaler2_b = preprocessing.MinMaxScaler()
        X_scaler2_c = preprocessing.MinMaxScaler()

        X_scaler3_a = preprocessing.MinMaxScaler()
        X_scaler3_b = preprocessing.MinMaxScaler()
        X_scaler3_c = preprocessing.MinMaxScaler()

        X_scaler4_a = preprocessing.MinMaxScaler()
        X_scaler4_b = preprocessing.MinMaxScaler()
        X_scaler4_c = preprocessing.MinMaxScaler()

        X_scaler5_a = preprocessing.MinMaxScaler()
        X_scaler5_b = preprocessing.MinMaxScaler()
        X_scaler5_c = preprocessing.MinMaxScaler()

        X_scaler6 = preprocessing.MinMaxScaler()
        X_scaler7 = preprocessing.MinMaxScaler()
        X_scaler8 = preprocessing.MinMaxScaler()
        X_scaler9 = preprocessing.MinMaxScaler()
        X_scaler10 = preprocessing.MinMaxScaler()

        Y_scaler = preprocessing.MinMaxScaler()

        rnn_scaled_train1_a = X_scaler1_a.fit_transform(rnn_train1_a)
        rnn_scaled_train1_b = X_scaler1_b.fit_transform(rnn_train1_b)
        rnn_scaled_train1_c = X_scaler1_c.fit_transform(rnn_train1_c)

        rnn_scaled_train2_a = X_scaler2_a.fit_transform(rnn_train2_a)
        rnn_scaled_train2_b = X_scaler2_b.fit_transform(rnn_train2_b)
        rnn_scaled_train2_c = X_scaler2_c.fit_transform(rnn_train2_c)

        rnn_scaled_train3_a = X_scaler3_a.fit_transform(rnn_train3_a)
        rnn_scaled_train3_b = X_scaler3_b.fit_transform(rnn_train3_b)
        rnn_scaled_train3_c = X_scaler3_c.fit_transform(rnn_train3_c)

        rnn_scaled_train4_a = X_scaler4_a.fit_transform(rnn_train4_a)
        rnn_scaled_train4_b = X_scaler4_b.fit_transform(rnn_train4_b)
        rnn_scaled_train4_c = X_scaler4_c.fit_transform(rnn_train4_c)

        rnn_scaled_train5_a = X_scaler5_a.fit_transform(rnn_train5_a)
        rnn_scaled_train5_b = X_scaler5_b.fit_transform(rnn_train5_b)
        rnn_scaled_train5_c = X_scaler5_c.fit_transform(rnn_train5_c)

        rnn_scaled_train6 = X_scaler6.fit_transform(rnn_train6)
        rnn_scaled_train7 = X_scaler7.fit_transform(rnn_train7)
        rnn_scaled_train8 = X_scaler8.fit_transform(rnn_train8)
        rnn_scaled_train9 = X_scaler9.fit_transform(rnn_train9)
        rnn_scaled_train10 = X_scaler10.fit_transform(rnn_train10)

        train_y = Y_scaler.fit_transform(rnn_Y)
        Y_scaler_n = Y_scaler.fit(rnn_Y)
        train_X = np.hstack(
            (rnn_scaled_train1_a, rnn_scaled_train1_b, rnn_scaled_train1_c, rnn_scaled_train2_a, rnn_scaled_train2_b,
             rnn_scaled_train2_c,
             rnn_scaled_train3_a, rnn_scaled_train3_b, rnn_scaled_train3_c, rnn_scaled_train4_a, rnn_scaled_train4_b,
             rnn_scaled_train4_c,
             rnn_scaled_train5_a, rnn_scaled_train5_b, rnn_scaled_train5_c, rnn_scaled_train6, rnn_scaled_train7,
             rnn_scaled_train8,
             rnn_scaled_train9, rnn_scaled_train10)
        ).reshape(rnn_train6.shape[0], 20, 16).transpose(0, 2, 1)

        test_X = np.hstack(
            (X_scaler1_a.transform(rnn_test1_a), X_scaler1_b.transform(rnn_test1_b), X_scaler1_c.transform(rnn_test1_c),
             X_scaler2_a.transform(rnn_test2_a), X_scaler2_b.transform(rnn_test2_b), X_scaler2_c.transform(rnn_test2_c),
             X_scaler3_a.transform(rnn_test3_a), X_scaler3_b.transform(rnn_test3_b), X_scaler3_c.transform(rnn_test3_c),
             X_scaler4_a.transform(rnn_test4_a), X_scaler4_b.transform(rnn_test4_b), X_scaler4_c.transform(rnn_test4_c),
             X_scaler5_a.transform(rnn_test5_a), X_scaler5_b.transform(rnn_test5_b), X_scaler5_c.transform(rnn_test5_c),
             X_scaler6.transform(rnn_test6), X_scaler7.transform(rnn_test7), X_scaler8.transform(rnn_test8),
             X_scaler9.transform(rnn_test9), X_scaler10.transform(rnn_test10))
        ).reshape(rnn_test6.shape[0], 20, 16).transpose(0, 2, 1)

        test_y = test_df.iloc[:, 0:16]

        return train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n


def fit_multitarget_model_SH_DNN(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets, Y_scaler_n):
    try:
        Y_scaler = preprocessing.MinMaxScaler()
        Y_scaler = Y_scaler.fit(Y_train)
        cols = Y.columns.values.tolist()

        model.fit(X_train, Y_train)
        model_test_predictions = None
        model_test_predictions = pd.DataFrame(Y_scaler_n.inverse_transform(model.predict(X_test).reshape(1, 16)),
                                              columns=cols, index=Y_test.index)
        model_test_mse = mean_squared_error(Y_test, model_test_predictions)
        model_test_rmse = round(np.sqrt(model_test_mse), 2)
        model_test_mae = round(mean_absolute_error(Y_test, model_test_predictions), 2)

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast"] = model_test_predictions.iloc[:, i].tolist() if len(
                cols) > 1 else model_test_predictions.tolist()
            predictor_test_mse = mean_squared_error(Y_test[cols[i]], model_test_predictions.iloc[:, i]) if len(
                cols) > 1 else mean_squared_error(Y_test[cols[i]], model_test_predictions.tolist())
            predictor_test_rmse = round(np.sqrt(predictor_test_mse), 2)
            predictor_test_mae = round(mean_absolute_error(Y_test[cols[i]], model_test_predictions.iloc[:, i]),
                                       2) if len(cols) > 1 else round(
                mean_absolute_error(Y_test[cols[i]], model_test_predictions.tolist()), 2)

        Error_i = ([model_test_rmse, model_test_mae])
        actuals_and_forecast_df = actuals_and_forecast_df.append(Error_i)

        return actuals_and_forecast_df

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()


def rolling_walk_forward_validation_SH_DNN(model, data, targets, start_time, end_time, training_days, path):
    try:

        all_columns = list(data.columns)
        results = pd.DataFrame()

        date_format = "%m/%d/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)

        while start_time < end_time:

            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time

            test_start_time = train_end_time + td(hours=8)
            test_end_time = test_start_time + td(minutes=30)

            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))

            train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n = generate_train_and_test_dataframes(
                participant_df=dat, train_start_time=train_start_time.strftime("%m/%d/%Y %H:%M"),
                train_end_time=train_end_time.strftime("%m/%d/%Y %H:%M"),
                test_start_time=test_start_time.strftime("%m/%d/%Y %H:%M"),
                test_end_time=test_end_time.strftime("%m/%d/%Y %H:%M"))

            if train_X is None or len(train_X) == 0:
                print("Don't have a train dataframe for train_start_time: " + str(
                    train_start_time) + ", train_end_time: " + str(train_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue

            if test_X is None or len(test_X) == 0:
                print("Don't have a test dataframe for test_start_time: " + str(
                    test_start_time) + ", test_end_time: " + str(test_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue

            actuals_and_forecast_df = fit_multitarget_model(model=model, Y_scaler_n=Y_scaler_n, X_train=train_X,
                                                            Y_train=train_y,
                                                            X_test=test_X, Y_test=test_y,
                                                            actuals_and_forecast_df=test_df.iloc[:, 0:16],
                                                            targets=Y.columns.values.tolist())

            results = results.append(actuals_and_forecast_df)

            start_time = start_time + td(minutes=30)

        results.to_csv(path + ".csv", index=False)



    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()


def create_model_SH_DNN():
    nn = Sequential()
    nn.add(Flatten(input_shape=i_shape))

    for i in range(2):
        nn.add(Dense(64, input_shape=i_shape, activation='tanh'))
        nn.add(BatchNormalization())

    for i in range(1):
        nn.add(Dense(128, activation='tanh'))
        nn.add(BatchNormalization())

    for i in range(1):
        nn.add(Dense(128, activation='relu'))
        nn.add(BatchNormalization())

    nn.add(Dropout(0.133333, seed=123))
    nn.add(BatchNormalization())

    for i in range(1):
        nn.add(Dense(128, activation='relu'))
        nn.add(BatchNormalization())

    nn.add(Dense(16, activation='relu'))
    nn.add(Dense(16))
    opt = Adam(lr=0.004522)
    nn.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

    return nn


def fit_multitarget_model_MH_DNN(model, X_train_LSTM, X_train_ffnn, Y_train, X_test_LSTM, X_test_ffnn, Y_test,
                          actuals_and_forecast_df, targets, Y_scaler_n):
    try:
        Y_scaler = preprocessing.MinMaxScaler()
        Y_scaler = Y_scaler.fit(Y_train)
        cols = Y.columns.values.tolist()

        model.fit([X_train_LSTM, X_train_ffnn], Y_train)
        model_test_predictions = None
        model_test_predictions = pd.DataFrame(
            Y_scaler_n.inverse_transform(model.predict([X_test_LSTM, X_test_ffnn]).reshape(1, 16)), columns=cols,
            index=Y_test.index)
        model_test_mse = mean_squared_error(Y_test, model_test_predictions)
        model_test_rmse = round(np.sqrt(model_test_mse), 2)
        model_test_mae = round(mean_absolute_error(Y_test, model_test_predictions), 2)

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast"] = model_test_predictions.iloc[:, i].tolist() if len(
                cols) > 1 else model_test_predictions.tolist()
            predictor_test_mse = mean_squared_error(Y_test[cols[i]], model_test_predictions.iloc[:, i]) if len(
                cols) > 1 else mean_squared_error(Y_test[cols[i]], model_test_predictions.tolist())
            predictor_test_rmse = round(np.sqrt(predictor_test_mse), 2)
            predictor_test_mae = round(mean_absolute_error(Y_test[cols[i]], model_test_predictions.iloc[:, i]),
                                       2) if len(cols) > 1 else round(
                mean_absolute_error(Y_test[cols[i]], model_test_predictions.tolist()), 2)

        Error_i = ([model_test_rmse, model_test_mae])
        actuals_and_forecast_df = actuals_and_forecast_df.append(Error_i)

        return actuals_and_forecast_df

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()


def rolling_walk_forward_validation_MH_DNN(model, data, targets, start_time, end_time, training_days, path):
    try:

        all_columns = list(data.columns)
        results = pd.DataFrame()

        date_format = "%m/%d/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)

        while start_time < end_time:

            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time

            test_start_time = train_end_time + td(hours=8)
            test_end_time = test_start_time + td(minutes=30)

            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))

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


def create_model_MH_DNN():
    visible1 = Input(shape=(i_shape_lstm))
    dense1 = LSTM(128, return_sequences=True, activation='tanh', input_shape=i_shape_lstm)(visible1)
    dense2 = LSTM(128, return_sequences=True, activation='tanh', input_shape=i_shape_lstm)(dense1)
    do_lstm = Dropout(0.044444, seed=123)(dense2)
    dense3 = LSTM(128)(do_lstm)
    flat1 = Flatten()(dense3)

    visible2 = Input(shape=(i_shape_ffnn))
    dense5 = Dense(16, activation='relu')(visible2)
    dense6 = Dense(16, activation=LeakyReLU)(dense5)
    do_ffnn = Dropout(0.200000, seed=123)(dense6)
    dense7 = Dense(16, activation='relu')(do_ffnn)
    flat2 = Flatten()(dense7)

    merged = concatenate([flat1, flat2])
    dense_f = Dense(256, activation='relu')(merged)
    outputs = Dense(16)(dense_f)

    model = Model(inputs=[visible1, visible2], outputs=outputs)
    opt = Adam(lr=0.004522)
    model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
    return model


def generate_train_and_test_dataframes_MH_DNN(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                       test_start_time: dt, test_end_time: dt):
    """
    This function generates training and testing dataframes for LSTM and FFNN models.

    Parameters:
    participant_df (pd.DataFrame): DataFrame containing participant data.
    train_start_time (dt): Start time for training data.
    train_end_time (dt): End time for training data.
    test_start_time (dt): Start time for testing data.
    test_end_time (dt): End time for testing data.

    Returns:
    train_X_LSTM (np.ndarray): Training features for LSTM model.
    train_X_ffnn (np.ndarray): Training features for FFNN model.
    train_y (np.ndarray): Training labels.
    test_X_LSTM (np.ndarray): Testing features for LSTM model.
    test_X_ffnn (np.ndarray): Testing features for FFNN model.
    test_y (pd.DataFrame): Testing labels.
    test_df (pd.DataFrame): DataFrame containing testing data.
    train_df (pd.DataFrame): DataFrame containing training data.
    Y_scaler_n (MinMaxScaler): Scaler for labels.
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

        original_columns = list(participant_df.columns)
        participant_df = participant_df.dropna()
        date_format = "%m/%d/%Y %H:%M"

        # Selecting data within specified time range for training and testing
        train_df = None
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
        train_df = participant_df[
            (participant_df.index >= train_start_time_str) & (participant_df.index < train_end_time_str)].copy(
            deep="True")

        if train_df is None or len(train_df) == 0:
            print(
                "Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")

        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format)
        test_df = participant_df[
            (participant_df.index >= test_start_time_str) & (participant_df.index < test_end_time_str)].copy(
            deep="True")

        if test_df is None or len(test_df) == 0:
            print(
                "Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")

        # Extracting lagged features for LSTM and FFNN models
        rnn_train_LSTM_1 = train_df.loc[:, "lag_-3x1":"lag_-50x1"]
        rnn_train_LSTM_2 = train_df.loc[:, "lag_-3x2": "lag_-50x2"]
        rnn_train_LSTM_3 = train_df.loc[:, "lag_-2x3":"lag_-49x3"]
        rnn_train_LSTM_4 = train_df.loc[:, "lag_0x6":"lag_-47x6"]
        rnn_train_LSTM_5 = train_df.loc[:, "lag_-2x12": "lag_-49x12"]

        rnn_test_LSTM_1 = test_df.loc[:, "lag_-3x1": "lag_-50x1"]
        rnn_test_LSTM_2 = test_df.loc[:, "lag_-3x2": "lag_-50x2"]
        rnn_test_LSTM_3 = test_df.loc[:, "lag_-2x3": "lag_-49x3"]
        rnn_test_LSTM_4 = test_df.loc[:, "lag_0x6": "lag_-47x6"]
        rnn_test_LSTM_5 = test_df.loc[:, "lag_-2x12": "lag_-49x12"]

        rnn_train_ffnn_1 = train_df.loc[:, "lag_2x7": "lag_17x7"]
        rnn_train_ffnn_2 = train_df.loc[:, "lag_2x8": "lag_17x8"]
        rnn_train_ffnn_3 = train_df.loc[:, "lag_2x9": "lag_17x9"]
        rnn_train_ffnn_4 = train_df.loc[:, "lag_2x10": "lag_17x10"]
        rnn_train_ffnn_5 = train_df.loc[:, "lag_2x11": "lag_17x11"]

        rnn_test_ffnn_1 = test_df.loc[:, "lag_2x7": "lag_17x7"]
        rnn_test_ffnn_2 = test_df.loc[:, "lag_2x8": "lag_17x8"]
        rnn_test_ffnn_3 = test_df.loc[:, "lag_2x9": "lag_17x9"]
        rnn_test_ffnn_4 = test_df.loc[:, "lag_2x10": "lag_17x10"]
        rnn_test_ffnn_5 = test_df.loc[:, "lag_2x11": "lag_17x11"]

        rnn_Y = train_df.loc[:, "lag_2y": "lag_17y"]

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

def qloss(qs, y_true, y_pred):
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


loss_10th_p = lambda y_true, y_pred: qloss(0.1, y_true, y_pred)
loss_30th_p = lambda y_true, y_pred: qloss(0.3, y_true, y_pred)
loss_50th_p = lambda y_true, y_pred: qloss(0.5, y_true, y_pred)
loss_70th_p = lambda y_true, y_pred: qloss(0.7, y_true, y_pred)
loss_90th_p = lambda y_true, y_pred: qloss(0.9, y_true, y_pred)



def prepare_data_MH_DNN(dat):
    """
    Preprocess the data and prepare it for LSTM and FFNN models.

    Parameters:
    dat (DataFrame): Input data.

    Returns:
    X_train_Scaled_LSTM (np.ndarray): Scaled training features for LSTM model.
    X_train_Scaled_ffnn (np.ndarray): Scaled training features for FFNN model.
    X_test_Scaled_LSTM (np.ndarray): Scaled testing features for LSTM model.
    X_test_Scaled_ffnn (np.ndarray): Scaled testing features for FFNN model.
    i_shape_lstm (tuple): Input shape for LSTM model.
    i_shape_ffnn (tuple): Input shape for FFNN model.
    """
    Y = dat.iloc[:, 0:16]  # Selecting the first 16 columns as labels
    X = dat.iloc[:, 16:]   # Selecting the columns from index 16 onwards as features

    # Splitting data into training and testing sets
    X_train = X.iloc[:7250, :]     # Training features
    Y_train = Y.iloc[:7250, :]     # Training labels
    X_test = X.iloc[7250:8739, :]  # Testing features
    Y_test = Y.iloc[7250:8739, :]  # Testing labels

    # Extracting specific lagged features for LSTM and FFNN models
    rnn_train_LSTM_1 = X_train.loc[:,"lag_-3x1":"lag_-50x1"]
    rnn_train_LSTM_2 = X_train.loc[:,"lag_-3x2": "lag_-50x2"]
    rnn_train_LSTM_3 = X_train.loc[:,"lag_-2x3":"lag_-49x3"]
    rnn_train_LSTM_4 = X_train.loc[:,"lag_0x6":"lag_-47x6"]
    rnn_train_LSTM_5 = X_train.loc[:,"lag_-2x12": "lag_-49x12"]

    rnn_test_LSTM_1 = X_test.loc[:,"lag_-3x1": "lag_-50x1"]
    rnn_test_LSTM_2 = X_test.loc[:,"lag_-3x2": "lag_-50x2"]
    rnn_test_LSTM_3 = X_test.loc[:,"lag_-2x3": "lag_-49x3"]
    rnn_test_LSTM_4 = X_test.loc[:,"lag_0x6" : "lag_-47x6"]
    rnn_test_LSTM_5 = X_test.loc[:,"lag_-2x12": "lag_-49x12"]

    rnn_train_ffnn_1 = X_train.loc[:,"lag_2x7": "lag_17x7"]
    rnn_train_ffnn_2 = X_train.loc[:,"lag_2x8": "lag_17x8"]
    rnn_train_ffnn_3 = X_train.loc[:,"lag_2x9": "lag_17x9"]
    rnn_train_ffnn_4 = X_train.loc[:,"lag_2x10": "lag_17x10"]
    rnn_train_ffnn_5 = X_train.loc[:,"lag_2x11": "lag_17x11"]

    rnn_test_ffnn_1 = X_test.loc[:,"lag_2x7":"lag_17x7"]
    rnn_test_ffnn_2 = X_test.loc[:,"lag_2x8": "lag_17x8"]
    rnn_test_ffnn_3 = X_test.loc[:,"lag_2x9": "lag_17x9"]
    rnn_test_ffnn_4 = X_test.loc[:,"lag_2x10":"lag_17x10"]
    rnn_test_ffnn_5 = X_test.loc[:,"lag_2x11": "lag_17x11"]

    rnn_Y = Y_train.loc[:,"lag_2y": "lag_17y"]

    # Scaling features and labels
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

    # Preparing data for LSTM and FFNN models
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

    return X_train_Scaled_LSTM, X_train_Scaled_ffnn, X_test_Scaled_LSTM, X_test_Scaled_ffnn, i_shape_lstm, i_shape_ffnn



def prepare_data_SH_DNN(dat):
    """
    Preprocess the data and prepare it for LSTM and FFNN models.

    Parameters:
    dat (DataFrame): Input data.

    Returns:
    X_train_Scaled (np.ndarray): Scaled training features.
    X_test_Scaled (np.ndarray): Scaled testing features.
    i_shape (tuple): Input shape.
    """
    Y = dat.iloc[:, 0:16]
    X = dat.iloc[:, 16:]
    X_train = X.iloc[:7250, :]
    Y_train = Y.iloc[:7250, :]
    X_test = X.iloc[7250:8739, :]
    Y_test = Y.iloc[7250:8739, :]

    rnn_train1_a = X_train.loc[:, "lag_-3x1":"lag_-18x1"]
    rnn_train1_b = X_train.loc[:, "lag_-19x1":"lag_-34x1"]
    rnn_train1_c = X_train.loc[:, "lag_-35x1":"lag_-50x1"]
    rnn_train2_a = X_train.loc[:, "lag_-3x2":"lag_-18x2"]
    rnn_train2_b = X_train.loc[:, "lag_-19x2":"lag_-34x2"]
    rnn_train2_c = X_train.loc[:, "lag_-35x2":"lag_-50x2"]
    rnn_train3_a = X_train.loc[:, "lag_-2x3":"lag_-17x3"]
    rnn_train3_b = X_train.loc[:, "lag_-18x3":"lag_-33x3"]
    rnn_train3_c = X_train.loc[:, "lag_-34x3":"lag_-49x3"]
    rnn_train4_a = X_train.loc[:, "lag_0x6":"lag_-15x6"]
    rnn_train4_b = X_train.loc[:, "lag_-16x6":"lag_-31x6"]
    rnn_train4_c = X_train.loc[:, "lag_-32x6":"lag_-47x6"]
    rnn_train5_a = X_train.loc[:, "lag_-2x12":"lag_-17x12"]
    rnn_train5_b = X_train.loc[:, "lag_-18x12":"lag_-33x12"]
    rnn_train5_c = X_train.loc[:, "lag_-34x12":"lag_-49x12"]
    rnn_train6 = X_train.loc[:, "lag_2x7":"lag_17x7"]
    rnn_train7 = X_train.loc[:, "lag_2x8":"lag_17x8"]
    rnn_train8 = X_train.loc[:, "lag_2x9":"lag_17x9"]
    rnn_train9 = X_train.loc[:, "lag_2x10":"lag_17x10"]
    rnn_train10 = X_train.loc[:, "lag_2x11":"lag_17x11"]

    rnn_test1_a = X_test.loc[:, "lag_-3x1":"lag_-18x1"]
    rnn_test1_b = X_test.loc[:, "lag_-19x1":"lag_-34x1"]
    rnn_test1_c = X_test.loc[:, "lag_-35x1":"lag_-50x1"]
    rnn_test2_a = X_test.loc[:, "lag_-3x2":"lag_-18x2"]
    rnn_test2_b = X_test.loc[:, "lag_-19x2":"lag_-34x2"]
    rnn_test2_c = X_test.loc[:, "lag_-35x2":"lag_-50x2"]
    rnn_test3_a = X_test.loc[:, "lag_-2x3":"lag_-17x3"]
    rnn_test3_b = X_test.loc[:, "lag_-18x3":"lag_-33x3"]
    rnn_test3_c = X_test.loc[:, "lag_-34x3":"lag_-49x3"]
    rnn_test4_a = X_test.loc[:, "lag_0x6":"lag_-15x6"]
    rnn_test4_b = X_test.loc[:, "lag_-16x6":"lag_-31x6"]
    rnn_test4_c = X_test.loc[:, "lag_-32x6":"lag_-47x6"]
    rnn_test5_a = X_test.loc[:, "lag_-2x12":"lag_-17x12"]
    rnn_test5_b = X_test.loc[:, "lag_-18x12":"lag_-33x12"]
    rnn_test5_c = X_test.loc[:, "lag_-34x12":"lag_-49x12"]
    rnn_test6 = X_test.loc[:, "lag_2x7":"lag_17x7"]
    rnn_test7 = X_test.loc[:, "lag_2x8":"lag_17x8"]
    rnn_test8 = X_test.loc[:, "lag_2x9":"lag_17x9"]
    rnn_test9 = X_test.loc[:, "lag_2x10":"lag_17x10"]
    rnn_test10 = X_test.loc[:, "lag_2x11":"lag_17x11"]

    rnn_Y = Y_train.loc[:, "lag_2y":"lag_17y"]

    X_scaler1_a = preprocessing.MinMaxScaler()
    X_scaler1_b = preprocessing.MinMaxScaler()
    X_scaler1_c = preprocessing.MinMaxScaler()

    X_scaler2_a = preprocessing.MinMaxScaler()
    X_scaler2_b = preprocessing.MinMaxScaler()
    X_scaler2_c = preprocessing.MinMaxScaler()

    X_scaler3_a = preprocessing.MinMaxScaler()
    X_scaler3_b = preprocessing.MinMaxScaler()
    X_scaler3_c = preprocessing.MinMaxScaler()

    X_scaler4_a = preprocessing.MinMaxScaler()
    X_scaler4_b = preprocessing.MinMaxScaler()
    X_scaler4_c = preprocessing.MinMaxScaler()

    X_scaler5_a = preprocessing.MinMaxScaler()
    X_scaler5_b = preprocessing.MinMaxScaler()
    X_scaler5_c = preprocessing.MinMaxScaler()

    X_scaler6 = preprocessing.MinMaxScaler()
    X_scaler7 = preprocessing.MinMaxScaler()
    X_scaler8 = preprocessing.MinMaxScaler()
    X_scaler9 = preprocessing.MinMaxScaler()
    X_scaler10 = preprocessing.MinMaxScaler()

    Y_scaler = preprocessing.MinMaxScaler()

    rnn_scaled_train1_a = X_scaler1_a.fit_transform(rnn_train1_a)
    rnn_scaled_train1_b = X_scaler1_b.fit_transform(rnn_train1_b)
    rnn_scaled_train1_c = X_scaler1_c.fit_transform(rnn_train1_c)

    rnn_scaled_train2_a = X_scaler2_a.fit_transform(rnn_train2_a)
    rnn_scaled_train2_b = X_scaler2_b.fit_transform(rnn_train2_b)
    rnn_scaled_train2_c = X_scaler2_c.fit_transform(rnn_train2_c)

    rnn_scaled_train3_a = X_scaler3_a.fit_transform(rnn_train3_a)
    rnn_scaled_train3_b = X_scaler3_b.fit_transform(rnn_train3_b)
    rnn_scaled_train3_c = X_scaler3_c.fit_transform(rnn_train3_c)

    rnn_scaled_train4_a = X_scaler4_a.fit_transform(rnn_train4_a)
    rnn_scaled_train4_b = X_scaler4_b.fit_transform(rnn_train4_b)
    rnn_scaled_train4_c = X_scaler4_c.fit_transform(rnn_train4_c)

    rnn_scaled_train5_a = X_scaler5_a.fit_transform(rnn_train5_a)
    rnn_scaled_train5_b = X_scaler5_b.fit_transform(rnn_train5_b)
    rnn_scaled_train5_c = X_scaler5_c.fit_transform(rnn_train5_c)

    rnn_scaled_train6 = X_scaler6.fit_transform(rnn_train6)
    rnn_scaled_train7 = X_scaler7.fit_transform(rnn_train7)
    rnn_scaled_train8 = X_scaler8.fit_transform(rnn_train8)
    rnn_scaled_train9 = X_scaler9.fit_transform(rnn_train9)
    rnn_scaled_train10 = X_scaler10.fit_transform(rnn_train10)

    Y_train_Scaled = Y_scaler.fit_transform(Y_train)

    X_train_Scaled = np.hstack(
        (rnn_scaled_train1_a, rnn_scaled_train1_b, rnn_scaled_train1_c, rnn_scaled_train2_a, rnn_scaled_train2_b,
         rnn_scaled_train2_c,
         rnn_scaled_train3_a, rnn_scaled_train3_b, rnn_scaled_train3_c, rnn_scaled_train4_a, rnn_scaled_train4_b,
         rnn_scaled_train4_c,
         rnn_scaled_train5_a, rnn_scaled_train5_b, rnn_scaled_train5_c, rnn_scaled_train6, rnn_scaled_train7,
         rnn_scaled_train8,
         rnn_scaled_train9, rnn_scaled_train10)
    ).reshape(rnn_train6.shape[0], 20, 16).transpose(0, 2, 1)

    X_test_Scaled = np.hstack(
        (X_scaler1_a.transform(rnn_test1_a), X_scaler1_b.transform(rnn_test1_b), X_scaler1_c.transform(rnn_test1_c),
         X_scaler2_a.transform(rnn_test2_a), X_scaler2_b.transform(rnn_test2_b), X_scaler2_c.transform(rnn_test2_c),
         X_scaler3_a.transform(rnn_test3_a), X_scaler3_b.transform(rnn_test3_b), X_scaler3_c.transform(rnn_test3_c),
         X_scaler4_a.transform(rnn_test4_a), X_scaler4_b.transform(rnn_test4_b), X_scaler4_c.transform(rnn_test4_c),
         X_scaler5_a.transform(rnn_test5_a), X_scaler5_b.transform(rnn_test5_b), X_scaler5_c.transform(rnn_test5_c),
         X_scaler6.transform(rnn_test6), X_scaler7.transform(rnn_test7), X_scaler8.transform(rnn_test8),
         X_scaler9.transform(rnn_test9), X_scaler10.transform(rnn_test10))
    ).reshape(rnn_test6.shape[0], 20, 16).transpose(0, 2, 1)

    i_shape = (X_train_Scaled.shape[1], X_train_Scaled.shape[2])

    return X_train_Scaled, X_test_Scaled, i_shape

# Usage example:
# X_train_Scaled, X_test_Scaled, i_shape = prepare_data(dat)


from sklearn.linear_model import LassoLarsIC


def prepare_data_LEAR(dat):
    """
    Preprocess the data and prepare it for LassoLarsIC model.

    Parameters:
    dat (DataFrame): Input data.

    Returns:
    alpha (float): Alpha value for LassoLarsIC model.
    """
    Y = dat.iloc[:, 0:16]
    X = dat.iloc[:, 16:]

    X_1 = X.loc[:, "lag_-3x1":"lag_-50x1"]
    X_2 = X.loc[:, "lag_-3x2":"lag_-50x2"]
    X_3 = X.loc[:, "lag_-2x3":"lag_-49x3"]
    X_4 = X.loc[:, "lag_0x6":"lag_-47x6"]
    X_5 = X.loc[:, "lag_-2x12":"lag_-49x12"]
    X_6 = X.loc[:, "lag_2x7":"lag_17x7"]
    X_7 = X.loc[:, "lag_2x8":"lag_17x8"]
    X_8 = X.loc[:, "lag_2x9":"lag_17x9"]
    X_9 = X.loc[:, "lag_2x10":"lag_17x10"]
    X_10 = X.loc[:, "lag_2x11":"lag_17x11"]

    X_test1 = X_test.loc[:, "lag_-3x1":"lag_-50x1"]
    X_test2 = X_test.loc[:, "lag_-3x2":"lag_-50x2"]
    X_test3 = X_test.loc[:, "lag_-2x3":"lag_-49x3"]
    X_test4 = X_test.loc[:, "lag_0x6":"lag_-47x6"]
    X_test5 = X_test.loc[:, "lag_-2x12":"lag_-49x12"]
    X_test6 = X_test.loc[:, "lag_2x7":"lag_17x7"]
    X_test7 = X_test.loc[:, "lag_2x8":"lag_17x8"]
    X_test8 = X_test.loc[:, "lag_2x9":"lag_17x9"]
    X_test9 = X_test.loc[:, "lag_2x10":"lag_17x10"]
    X_test10 = X_test.loc[:, "lag_2x11":"lag_17x11"]
    Y_1 = Y_train.loc[:, "lag_2y": "lag_17y"]

    [X_1], X_scaler1 = scaling([X_1.values], 'Invariant')
    [X_2], X_scaler2 = scaling([X_2.values], 'Invariant')
    [X_3], X_scaler3 = scaling([X_3.values], 'Invariant')
    [X_4], X_scaler4 = scaling([X_4.values], 'Invariant')
    [X_5], X_scaler5 = scaling([X_5.values], 'Invariant')

    [X_6], X_scaler6 = scaling([X_6.values], 'Invariant')
    [X_7], X_scaler7 = scaling([X_7.values], 'Invariant')
    [X_8], X_scaler8 = scaling([X_8.values], 'Invariant')
    [X_9], X_scaler9 = scaling([X_9.values], 'Invariant')
    [X_10], X_scaler10 = scaling([X_10.values], 'Invariant')

    X_test_1 = X_scaler1.transform(X_test1.values)
    X_test_2 = X_scaler2.transform(X_test2.values)
    X_test_3 = X_scaler3.transform(X_test3.values)
    X_test_4 = X_scaler4.transform(X_test4.values)
    X_test_5 = X_scaler5.transform(X_test5.values)
    X_test_6 = X_scaler6.transform(X_test6.values)
    X_test_7 = X_scaler7.transform(X_test7.values)
    X_test_8 = X_scaler8.transform(X_test8.values)
    X_test_9 = X_scaler9.transform(X_test9.values)
    X_test_10 = X_scaler10.transform(X_test10.values)

    [Y_train_scaled], Y_scaler = scaling([Y_1.values], 'Invariant')

    X_train_scaled = np.concatenate((X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10), axis=1)
    X_test_scaled = np.concatenate(
        (X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6, X_test_7, X_test_8, X_test_9, X_test_10), axis=1)

    alpha = LassoLarsIC(criterion='aic', max_iter=2500).fit(X_train_scaled, Y_train_scaled[:, :1].ravel()).alpha_

    return alpha

def generate_train_and_test_dataframes_LEAR(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                       test_start_time: dt, test_end_time: dt):


    train_X = None
    train_Y = None
    test_X = None
    test_Y = None
    test_df = None
    train_df = None

    try:

        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
        #             return train_X, train_y, test_X, test_y, test_df

        original_columns = list(participant_df.columns)

        participant_df = participant_df.dropna()

        date_format = "%m/%d/%Y %H:%M"

        train_df = None
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
        train_df = participant_df[
            (participant_df.index >= train_start_time_str) & (participant_df.index < train_end_time_str)].copy(
            deep="True")

        if train_df is None or len(train_df) == 0:
            print(
                "Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")

        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format)
        test_df = participant_df[
            (participant_df.index >= test_start_time_str) & (participant_df.index < test_end_time_str)].copy(
            deep="True")

        if test_df is None or len(test_df) == 0:
            print(
                "Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")

        X_1 = train_df.loc[:,"lag_-3x1":"lag_-50x1"]
        X_2 = train_df.loc[:,"lag_-3x2":"lag_-50x2"]
        X_3 = train_df.loc["lag_-2x3":"lag_-49x3"]
        X_4 = train_df.loc[:,"lag_0x6":"lag_-47x6"]
        X_5 = train_df.loc[:,"lag_-2x12":"lag_-49x12"]
        X_6 = train_df.loc[:,"lag_2x7":"lag_17x7"]
        X_7 = train_df.loc[:,"lag_2x8":"lag_17x8"]
        X_8 = train_df.loc[:,"lag_2x9":"lag_17x9"]
        X_9 = train_df.loc[:,"lag_2x10":"lag_17x10"]
        X_10 = train_df.loc[:,"lag_2x11": "lag_17x11"]

        X_test1 = test_df.loc[:,"lag_-3x1":"lag_-50x1"]
        X_test2 = test_df.loc[:,"lag_-3x2":"lag_-50x2"]
        X_test3 = test_df.loc[:,"lag_-2x3":"lag_-49x3"]
        X_test4 = test_df.loc[:,"lag_0x6":"lag_-47x6"]
        X_test5 = test_df.loc[:,"lag_-2x12":"lag_-49x12"]
        X_test6 = test_df.loc[:,"lag_2x7":"lag_17x7"]
        X_test7 = test_df.loc[:,"lag_2x8":"lag_17x8"]
        X_test8 = test_df.loc[:,"lag_2x9":"lag_17x9"]
        X_test9 = test_df.loc[:,"lag_2x10":"lag_17x10"]
        X_test10 = test_df.loc[:,"lag_2x11":"lag_17x11"]
        Y_1 = train_df.loc[:,"lag_2y": "lag_17y"]

        [X_1], X_scaler1 = scaling([X_1.values], 'Invariant')
        [X_2], X_scaler2 = scaling([X_2.values], 'Invariant')
        [X_3], X_scaler3 = scaling([X_3.values], 'Invariant')
        [X_4], X_scaler4 = scaling([X_4.values], 'Invariant')
        [X_5], X_scaler5 = scaling([X_5.values], 'Invariant')

        [X_6], X_scaler6 = scaling([X_6.values], 'Invariant')
        [X_7], X_scaler7 = scaling([X_7.values], 'Invariant')
        [X_8], X_scaler8 = scaling([X_8.values], 'Invariant')
        [X_9], X_scaler9 = scaling([X_9.values], 'Invariant')
        [X_10], X_scaler10 = scaling([X_10.values], 'Invariant')

        X_test_1= X_scaler1.transform(X_test1.values)
        X_test_2= X_scaler2.transform(X_test2.values)
        X_test_3= X_scaler3.transform(X_test3.values)
        X_test_4= X_scaler4.transform(X_test4.values)
        X_test_5= X_scaler5.transform(X_test5.values)
        X_test_6= X_scaler6.transform(X_test6.values)
        X_test_7= X_scaler7.transform(X_test7.values)
        X_test_8= X_scaler8.transform(X_test8.values)
        X_test_9= X_scaler9.transform(X_test9.values)
        X_test_10= X_scaler10.transform(X_test10.values)

        [train_Y], Y_scaler = scaling([Y_1.values], 'Invariant')
        Y_scaler_n = Y_scaler

        train_X=np.concatenate((X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10), axis=1)
        test_X=np.concatenate((X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6, X_test_7, X_test_8, X_test_9, X_test_10), axis=1)
        test_Y = test_df.iloc[:, 0:16]

        return train_X, train_Y, test_X, test_Y, test_df, train_df, Y_scaler_n

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_Y, test_X, test_Y, test_df, train_df, Y_scaler_n


def fit_multitarget_model_LEAR(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets, Y_scaler_n):
    try:
        cols = Y.columns.values.tolist()

        model.fit(X_train, Y_train)
        model_test_predictions = None
        model_test_predictions = pd.DataFrame(Y_scaler_n.inverse_transform(model.predict(X_test)))
        model_test_mse = mean_squared_error(Y_test, model_test_predictions)
        model_test_rmse = round(np.sqrt(model_test_mse), 2)
        model_test_mae = round(mean_absolute_error(Y_test, model_test_predictions), 2)

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast"] = model_test_predictions.iloc[:, i].tolist() if len(
                cols) > 1 else model_test_predictions.tolist()
            predictor_test_mse = mean_squared_error(Y_test[cols[i]], model_test_predictions.iloc[:, i]) if len(
                cols) > 1 else mean_squared_error(Y_test[cols[i]], model_test_predictions.tolist())
            predictor_test_rmse = round(np.sqrt(predictor_test_mse), 2)
            predictor_test_mae = round(mean_absolute_error(Y_test[cols[i]], model_test_predictions.iloc[:, i]),
                                       2) if len(cols) > 1 else round(
                mean_absolute_error(Y_test[cols[i]], model_test_predictions.tolist()), 2)

        Error_i = ([model_test_rmse, model_test_mae])
        actuals_and_forecast_df = actuals_and_forecast_df.append(Error_i)

        return actuals_and_forecast_df

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()




def rolling_walk_forward_validation_LEAR(model, data, targets, start_time, end_time, training_days, path):
    try:

        all_columns = list(data.columns)
        results = pd.DataFrame()

        date_format = "%m/%d/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)

        while start_time < end_time:

            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time

            test_start_time = train_end_time + td(hours=8)
            test_end_time = test_start_time + td(minutes=30)

            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))

            train_X, train_Y, test_X, test_Y, test_df, train_df, Y_scaler_n = generate_train_and_test_dataframes(
                participant_df=dat, train_start_time=train_start_time.strftime("%m/%d/%Y %H:%M"),
                train_end_time=train_end_time.strftime("%m/%d/%Y %H:%M"),
                test_start_time=test_start_time.strftime("%m/%d/%Y %H:%M"),
                test_end_time=test_end_time.strftime("%m/%d/%Y %H:%M"))

            if train_X is None or len(train_X) == 0:
                print("Don't have a train dataframe for train_start_time: " + str(
                    train_start_time) + ", train_end_time: " + str(train_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue

            if test_X is None or len(test_X) == 0:
                print("Don't have a test dataframe for test_start_time: " + str(
                    test_start_time) + ", test_end_time: " + str(test_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue

            actuals_and_forecast_df = fit_multitarget_model(model=model, Y_scaler_n=Y_scaler_n, X_train=train_X,
                                                            Y_train=train_Y, X_test=test_X, Y_test=test_Y,
                                                            actuals_and_forecast_df=test_df.iloc[:, 0:16],
                                                            targets=Y.columns.values.tolist())

            results = results.append(actuals_and_forecast_df)

            start_time = start_time + td(minutes=30)

        results.to_csv(path + ".csv", index=False)


    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()

# Function to generate train and test dataframes
def generate_train_and_test_dataframes_ARIMA(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                               test_start_time: dt, test_end_time: dt):
            """
            This function generates training and testing dataframes for LSTM and FFNN models.

            Parameters:
            participant_df (pd.DataFrame): DataFrame containing participant data.
            train_start_time (dt): Start time for training data.
            train_end_time (dt): End time for training data.
            test_start_time (dt): Start time for testing data.
            test_end_time (dt): End time for testing data.

            Returns:
            train_X_LSTM (np.ndarray): Training features for LSTM model.
            train_X_ffnn (np.ndarray): Training features for FFNN model.
            train_y (np.ndarray): Training labels.
            test_X_LSTM (np.ndarray): Testing features for LSTM model.
            test_X_ffnn (np.ndarray): Testing features for FFNN model.
            test_y (pd.DataFrame): Testing labels.
            test_df (pd.DataFrame): DataFrame containing testing data.
            train_df (pd.DataFrame): DataFrame containing training data.
            Y_scaler_n (MinMaxScaler): Scaler for labels.
            """
            # These are the dataframes that will be returned from the method.

            train_X = None
            train_y = None
            test_X = None
            test_y = None
            test_df = None

            try:

                if len(participant_df) == 0:
                    print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
                    return train_X, train_y, test_X, test_y, test_df

                original_columns = list(participant_df.columns)

                participant_df = participant_df.dropna()

                date_format = "%m/%d/%Y %H:%M"

                # Selecting data within specified time range for training and testing
                train_df = None

                train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
                train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
                train_df = participant_df[
                    (participant_df.index >= train_start_time_str) & (participant_df.index < train_end_time_str)].copy(
                    deep="True")

                if train_df is None or len(train_df) == 0:
                    print(
                        "Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")
                    return train_X, train_y, test_X, test_y, test_df

                test_start_time_str = dt.datetime.strptime(test_start_time, date_format)
                test_end_time_str = dt.datetime.strptime(test_end_time, date_format)
                test_df = participant_df[
                    (participant_df.index >= test_start_time_str) & (participant_df.index < test_end_time_str)].copy(
                    deep="True")

                if test_df is None or len(test_df) == 0:
                    print(
                        "Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")
                    return train_X, train_y, test_X, test_y, test_df

                train_X = train_df.iloc[:, 16:]
                test_X = test_df.iloc[:, 16:]
                train_y = train_df.iloc[:, 0:1]
                test_y = test_df.iloc[:, 0:1]

                return train_X, train_y, test_X, test_y, test_df

            except Exception:
                print("Error: generate_train_and_test_dataframes method.")
                traceback.print_exc()
                return train_X, train_y, test_X, test_y, test_df

def fit_multitarget_model_ARIMA(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets):
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
                model_test_predictions = np.zeros((X_test.shape[0], len(model)))
                for i, mod in enumerate(model):
                    model_test_predictions[:, i] = mod.forecast(steps=X_test.shape[0])
                cols = Y_train.columns.values.tolist()

                for i in range(0, len(cols)):
                    actuals_and_forecast_df[cols[i] + "_Forecast"] = model_test_predictions[:, i].tolist() if len(
                        cols) > 1 else model_test_predictions.tolist()

                return actuals_and_forecast_df
                print(actuals_and_forecast_df)

            except Exception:
                print("Error: fit_multitarget_model method.")
                traceback.print_exc()
                return pd.DataFrame()

def rolling_walk_forward_validation_ARIMA(model_fn, data, start_time, end_time, training_days, path, targets):
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
                results = pd.DataFrame()
                start_time = datetime.strptime(start_time, "%m/%d/%Y %H:%M")
                end_time = datetime.strptime(end_time, "%m/%d/%Y %H:%M")

                while start_time < end_time:
                    train_start_time = start_time - timedelta(days=training_days)
                    train_end_time = start_time

                    test_start_time = train_end_time
                    test_end_time = train_end_time + timedelta(hours=8)

                    train_X = data[(data.index >= train_start_time) & (data.index < train_end_time)]
                    test_X = data[(data.index >= test_start_time) & (data.index < test_end_time)]

                    if train_X.empty or test_X.empty:
                        start_time += timedelta(hours=8)
                        continue

                    model = model_fn(train_X.iloc[:, 16:], train_X.iloc[:, :16])
                    model_test_predictions = [m.predict(test_X.iloc[:, 16:]) for m in model]

                    actuals_and_forecast_df = pd.DataFrame(model_test_predictions).T
                    actuals_and_forecast_df.columns = [f"{col}_Forecast" for col in targets]

                    results = pd.concat([results, actuals_and_forecast_df])

                    start_time += timedelta(hours=8)

                results.to_csv(path + ".csv", index=False)

            except Exception:
                print("Error: rolling_walk_forward_validation method.")
                traceback.print_exc()

# Function to generate train and test dataframes
def generate_train_and_test_dataframes_ARIMA(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                       test_start_time: dt, test_end_time: dt):
    """
    This function generates training and testing dataframes for LSTM and FFNN models.

    Parameters:
    participant_df (pd.DataFrame): DataFrame containing participant data.
    train_start_time (dt): Start time for training data.
    train_end_time (dt): End time for training data.
    test_start_time (dt): Start time for testing data.
    test_end_time (dt): End time for testing data.

    Returns:
    train_X_LSTM (np.ndarray): Training features for LSTM model.
    train_X_ffnn (np.ndarray): Training features for FFNN model.
    train_y (np.ndarray): Training labels.
    test_X_LSTM (np.ndarray): Testing features for LSTM model.
    test_X_ffnn (np.ndarray): Testing features for FFNN model.
    test_y (pd.DataFrame): Testing labels.
    test_df (pd.DataFrame): DataFrame containing testing data.
    train_df (pd.DataFrame): DataFrame containing training data.
    Y_scaler_n (MinMaxScaler): Scaler for labels.
    """
    # These are the dataframes that will be returned from the method.

    train_X = None
    train_y = None
    test_X = None
    test_y = None
    test_df = None

    try:

        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
            return train_X, train_y, test_X, test_y, test_df

        original_columns = list(participant_df.columns)

        participant_df = participant_df.dropna()

        date_format = "%m/%d/%Y %H:%M"

        # Selecting data within specified time range for training and testing
        train_df = None

        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
        train_df = participant_df[
            (participant_df.index >= train_start_time_str) & (participant_df.index < train_end_time_str)].copy(
            deep="True")


        if train_df is None or len(train_df) == 0:
            print(
                "Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")
            return train_X, train_y, test_X, test_y, test_df

        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format)
        test_df = participant_df[
            (participant_df.index >= test_start_time_str) & (participant_df.index < test_end_time_str)].copy(
            deep="True")

        if test_df is None or len(test_df) == 0:
            print(
                "Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")
            return train_X, train_y, test_X, test_y, test_df

        train_X = train_df.iloc[:, 16:]
        test_X = test_df.iloc[:, 16:]
        train_y = train_df.iloc[:, 0:1]
        test_y = test_df.iloc[:, 0:1]

        return train_X, train_y, test_X, test_y, test_df

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df


def fit_multitarget_model_ARIMA(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets):
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
        model_test_predictions = np.zeros((X_test.shape[0], len(model)))
        for i, mod in enumerate(model):
            model_test_predictions[:, i] = mod.forecast(steps=X_test.shape[0])
        cols = Y_train.columns.values.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast"] = model_test_predictions[:, i].tolist() if len(
                cols) > 1 else model_test_predictions.tolist()

        return actuals_and_forecast_df
        print(actuals_and_forecast_df)

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()


def rolling_walk_forward_validation_ARIMA(model_fn, data, start_time, end_time, training_days, path, targets):
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
        results = pd.DataFrame()
        start_time = datetime.strptime(start_time, "%m/%d/%Y %H:%M")
        end_time = datetime.strptime(end_time, "%m/%d/%Y %H:%M")

        while start_time < end_time:
            train_start_time = start_time - timedelta(days=training_days)
            train_end_time = start_time

            test_start_time = train_end_time
            test_end_time = train_end_time + timedelta(hours=8)

            train_X = data[(data.index >= train_start_time) & (data.index < train_end_time)]
            test_X = data[(data.index >= test_start_time) & (data.index < test_end_time)]

            if train_X.empty or test_X.empty:
                start_time += timedelta(hours=8)
                continue

            model = model_fn(train_X.iloc[:, 16:], train_X.iloc[:, :16])
            model_test_predictions = [m.predict(test_X.iloc[:, 16:]) for m in model]

            actuals_and_forecast_df = pd.DataFrame(model_test_predictions).T
            actuals_and_forecast_df.columns = [f"{col}_Forecast" for col in targets]

            results = pd.concat([results, actuals_and_forecast_df])

            start_time += timedelta(hours=8)

        results.to_csv(path + ".csv", index=False)

    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()


from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestQuantileRegressor
from sklearn.svm import SVR
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Flatten, concatenate, LeakyReLU
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def rolling_walk_forward_validation(model, data, start_time, end_time, targets, training_days, path):
    """
    Perform rolling walk forward validation for different models.

    Parameters:
    model: str
        Name of the model to be used for the rolling walk forward validation.
    data: DataFrame
        Input data.
    start_time: str
        Start time for the validation period.
    end_time: str
        End time for the validation period.
    targets: list
        List of target column names.
    training_days: int
        Number of days for training.
    path: str
        Path to save the results.

    Returns:
    None
    """
    # Model selection based on the specified model name
    if model == "XGB":
        # Define XGB model
        model_instance = MultiOutputRegressor(XGBRegressor(learning_rate=0.05, max_depth=2, n_estimators=50))
    elif model == "SH_DNN":
        # Define SH DNN model
        def create_sh_dnn_model():
            nn = Sequential()
            nn.add(Flatten(input_shape=i_shape))

            for i in range(2):
                nn.add(Dense(64, input_shape=i_shape, activation='tanh'))
                nn.add(BatchNormalization())

            for i in range(1):
                nn.add(Dense(128, activation='tanh'))
                nn.add(BatchNormalization())

            for i in range(1):
                nn.add(Dense(128, activation='relu'))
                nn.add(BatchNormalization())

            nn.add(Dropout(0.133333, seed=123))
            nn.add(BatchNormalization())

            for i in range(1):
                nn.add(Dense(128, activation='relu'))
                nn.add(BatchNormalization())

            nn.add(Dense(16, activation=LeakyReLU))
            nn.add(Dense(16))
            opt = Adam(lr=0.004522)
            nn.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

            return nn


        es = EarlyStopping(monitor='mean_absolute_error', mode='min', verbose=0, patience=20)
        model_instance = KerasRegressor(build_fn=create_sh_dnn_model, epochs=300, batch_size=16, verbose=2,
                                        callbacks=[es])
    elif model == "RF":
        # Define RF model
        model_instance = MultiOutputRegressor(
            RandomForestQuantileRegressor(q=[0.50], max_depth=2, min_samples_leaf=2, n_estimators=100,
                                          min_samples_split=2))
    elif model == "SVR":
        # Define SVR model
        model_instance = MultiOutputRegressor(SVR(C=0.01, gamma=0.0001, epsilon=0.1, kernel='rbf'))
    elif model == "LEAR":
        # Define LEAR model
        alpha =  # Calculate alpha for LassoLarsIC
        model_instance = MultiOutputRegressor(Lasso(max_iter=2500, alpha=alpha))
    elif model == "MH_DNN":
        # Define MH DNN model
        def create_mh_dnn_model():
            visible1 = Input(shape=(i_shape_lstm))
            dense1 = LSTM(128, return_sequences=True, activation='tanh', input_shape=i_shape_lstm)(visible1)
            dense2 = LSTM(128, return_sequences=True, activation='tanh', input_shape=i_shape_lstm)(dense1)
            do_lstm = Dropout(0.044444, seed=123)(dense2)
            dense3 = LSTM(128)(do_lstm)
            flat1 = Flatten()(dense3)

            visible2 = Input(shape=(i_shape_ffnn))
            dense5 = Dense(16, activation='relu')(visible2)
            dense6 = Dense(16, activation=LeakyReLU)(dense5)
            do_ffnn = Dropout(0.200000, seed=123)(dense6)
            dense7 = Dense(16, activation='relu')(do_ffnn)
            flat2 = Flatten()(dense7)

            merged = concatenate([flat1, flat2])
            dense_f = Dense(256, activation='relu')(merged)
            outputs = Dense(16)(dense_f)

            model = Model(inputs=[visible1, visible2], outputs=outputs)
            opt = Adam(lr=0.004522)
            model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
            return nn

        es = EarlyStopping(monitor='mean_absolute_error', mode='min', verbose=0, patience=20)
        model_instance = KerasRegressor(build_fn=create_mh_dnn_model, epochs=500, batch_size=48, verbose=2,
                                        callbacks=[es])
    else:
        raise ValueError("Invalid model name")


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, make_scorer
from talos.utils import early_stopper
import talos
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
import numpy as np


def preprocess_data_hyp_dnn(dat, train_start, train_end, test_start, test_end):
    Y = dat.iloc[:, 0:16]
    X = dat.iloc[:, 16:]

    X_train = X.loc[train_start:train_end]
    Y_train = Y.loc[train_start:train_end]

    X_test = X.loc[test_start:test_end]
    Y_test = Y.loc[test_start:test_end]

    # Preprocessing steps...
    # Combine all preprocessing steps here...

    return X_train_Scaled, Y_train_Scaled, X_test_Scaled, Y_test_scaled


def create_model_hyp_dnn(x_train, y_train, x_val, y_val, params):
    nn = Sequential()
    nn.add(Flatten(input_shape=i_shape))

    for i in range(params['layers1']):
        nn.add(Dense(params['neurons_0'], input_shape=i_shape, activation=params['activation_0']))

    for i in range(params['layers2']):
        nn.add(Dense(params['neurons_1'], activation=params['activation_1']))

    for i in range(params['layers3']):
        nn.add(Dense(params['neurons_2'], activation=params['activation_2']))

    nn.add(Dropout(params['dropout_rate'], seed=123))

    for i in range(params['layers4']):
        nn.add(Dense(params['neurons_3'], activation=params['activation_3']))

    nn.add(Dense(params['neurons_4'], activation=params['activation_4']))
    nn.add(Dense(16))
    opt = Adam(lr=params['learning_rate'])
    nn.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
    out = nn.fit(x_train, y_train, validation_data=[x_val, y_val], batch_size=params['batch_size'],
                 epochs=params['epochs'], verbose=2,
                 callbacks=[early_stopper(epochs=params['epochs'], mode='moderate', min_delta=0.001,
                                          monitor='mean_absolute_error')])
    return out, nn


def perform_hyperparameter_search(dat, train_start, train_end, test_start, test_end, model_name):
    X_train_Scaled, Y_train_Scaled, X_test_Scaled, Y_test_scaled = preprocess_data_hyp_dnn(dat, train_start, train_end,
                                                                                   test_start, test_end)

    pp = {
            'neurons_0': list(range(16, 256, 16)),
            'activation_0':['relu', 'sigmoid',  'tanh', LeakyReLU],
            'neurons_1': list(range(16, 256, 16)),
            'activation_1':['relu', 'sigmoid',  'tanh', LeakyReLU],
            'neurons_2': list(range(16, 256, 16)),
            'activation_2':['relu', 'sigmoid',  'tanh', LeakyReLU],
            'neurons_3': list(range(16, 256, 16)),
            'activation_3':['relu', 'sigmoid',  'tanh', LeakyReLU],
            'neurons_4': list(range(16, 256, 16)),
            'activation_4':['relu', 'sigmoid',  'tanh', LeakyReLU],

            'learning_rate': list(np.linspace(0.0001,0.02, 10)),
            'layers1': list(range(1,3, 1)),
            'layers2': list(range(1,3, 1)),
            'layers3': list(range(1,3, 1)),
            'layers4': list(range(1,3, 1)),
            'dropout_rate': list(np.linspace(0.0,0.2, 10)),
            'batch_size': list(range(16, 64, 16)),
            'epochs': [200]
        }

    h = talos.Scan(x=X_train_Scaled, y=Y_train_Scaled, x_val=X_test_Scaled, y_val=Y_test_scaled, params=pp,
                   model=create_model_hyp_dnn, val_split=0.2,
                   experiment_name='bm_1-3', random_method='quantum', round_limit=30, print_params=True)

    h.data.sort_values(by='val_mean_absolute_error', ascending=True)


# Example of using the function for different models
# perform_hyperparameter_search(dat, '06/1/2020', '06/6/2021', '06/1/2020', '06/6/2021', model_name="SH_DNN")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, make_scorer
from talos.utils import early_stopper
import talos
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Flatten, Input, concatenate
from keras.optimizers import Adam
import numpy as np

def perform_hyperparameter_search_MH_DNN(dat, train_start, train_end, test_start, test_end):
    def preprocess_data_MH_DNN(dat, train_start, train_end, test_start, test_end):
        Y = dat.iloc[:, 0:16]
        X = dat.iloc[:, 16:]

        X_train = X.loc[train_start:train_end]
        Y_train = Y.loc[train_start:train_end]

        X_test = X.loc[test_start:test_end]
        Y_test = Y.loc[test_start:test_end]

        rnn_train_LSTM_1 = X_train.loc[:,"lag_-3x1":"lag_-50x1"]
        rnn_train_LSTM_2 = X_train.loc[:,"lag_-3x2": "lag_-50x2"]
        rnn_train_LSTM_3 = X_train.loc[:,"lag_-2x3":"lag_-49x3"]
        rnn_train_LSTM_4 = X_train.loc[:,"lag_0x6":"lag_-47x6"]
        rnn_train_LSTM_5 = X_train.loc[:,"lag_-2x12": "lag_-49x12"]

        rnn_test_LSTM_1 = X_test.loc[:,"lag_-3x1": "lag_-50x1"]
        rnn_test_LSTM_2 = X_test.loc[:,"lag_-3x2": "lag_-50x2"]
        rnn_test_LSTM_3 = X_test.loc[:,"lag_-2x3": "lag_-49x3"]
        rnn_test_LSTM_4 = X_test.loc[:,"lag_0x6" : "lag_-47x6"]
        rnn_test_LSTM_5 = X_test.loc[:,"lag_-2x12": "lag_-49x12"]

        rnn_train_ffnn_1 = X_train.loc[:,"lag_2x7": "lag_17x7"]
        rnn_train_ffnn_2 = X_train.loc[:,"lag_2x8": "lag_17x8"]
        rnn_train_ffnn_3 = X_train.loc[:,"lag_2x9": "lag_17x9"]
        rnn_train_ffnn_4 = X_train.loc[:,"lag_2x10": "lag_17x10"]
        rnn_train_ffnn_5 = X_train.loc[:,"lag_2x11": "lag_17x11"]

        rnn_test_ffnn_1 = X_test.loc[:,"lag_2x7":"lag_17x7"]
        rnn_test_ffnn_2 = X_test.loc[:,"lag_2x8": "lag_17x8"]
        rnn_test_ffnn_3 = X_test.loc[:,"lag_2x9": "lag_17x9"]
        rnn_test_ffnn_4 = X_test.loc[:,"lag_2x10":"lag_17x10"]
        rnn_test_ffnn_5 = X_test.loc[:,"lag_2x11": "lag_17x11"]

        Y_scaler = MinMaxScaler()

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

        return X_train_Scaled_LSTM, X_train_Scaled_ffnn, Y_train_Scaled, X_test_Scaled_LSTM, X_test_Scaled_ffnn, Y_test_scaled

    def create_model_MH_DNN(x_train, y_train, x_val, y_val, params):
        i_shape_lstm = (x_train[0].shape[1], x_train[0].shape[2])
        i_shape_ffnn = (x_train[1].shape[1], x_train[1].shape[2])

        visible1 = Input(shape=i_shape_lstm)
        dense1 = LSTM(params['lstm_neurons_0'], return_sequences=True, activation=params['lstm_activation_0'])(visible1)
        dense2 = LSTM(params['lstm_neurons_0'], return_sequences=True, activation=params['lstm_activation_0'])(dense1)
        do_lstm = Dropout(params['dropout_rate_lstm'])(dense2)
        dense3 = LSTM(params['lstm_neurons_0'], return_sequences=True, activation=params['lstm_activation_0'])(do_lstm)
        flat1 = Flatten()(dense3)

        visible2 = Input(shape=i_shape_ffnn)
        dense5 = Dense(params['ffnn_neurons_0'], activation=params['activation_0'])(visible2)
        dense6 = Dense(params['ffnn_neurons_0'], activation=params['activation_0'])(dense5)
        do_ffnn = Dropout(params['dropout_rate_ffnn'])(dense6)
        dense7 = Dense(params['ffnn_neurons_0'], activation=params['activation_0'])(do_ffnn)
        flat2 = Flatten()(dense7)

        merged = concatenate([flat1, flat2])
        dense_f = Dense(params['dense_f_neurons'], activation=params['activation_1'])(merged)
        output = Dense(16)(dense_f)
        model = Model(inputs=[visible1, visible2], outputs=output)

        opt = Adam(lr=params['learning_rate'])
        model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])

        out = model.fit(x=x_train, y=y_train, validation_data=[x_val, y_val],
                        batch_size=params['batch_size'], epochs=params['epochs'], verbose=2,
                        callbacks=[early_stopper(epochs=params['epochs'], mode='moderate', min_delta=0.001, monitor='mean_absolute_error')])
        return out, model

    pp = {
        'lstm_neurons_0': [16, 32, 64, 128, 192, 256],
        'ffnn_neurons_0': [16, 32, 64, 128, 192, 256],
        'dense_f_neurons': [16, 32, 64, 128, 192, 256],
        'lstm_activation_0': ['relu', 'sigmoid', 'tanh'],
        'activation_0': ['relu', 'sigmoid', 'tanh'],
        'dropout_rate_lstm': [0.0, 0.1, 0.2],
        'dropout_rate_ffnn': [0.0, 0.1, 0.2],
        'learning_rate': list(np.linspace(0.0001, 0.02, 10)),
        'batch_size': [4, 8, 16, 32, 48],
        'epochs': [300]
    }

    X_train_Scaled_LSTM, X_train_Scaled_ffnn, Y_train_Scaled, X_test_Scaled_LSTM, X_test_Scaled_ffnn, Y_test_scaled = preprocess_data_MH_DNN(dat, train_start, train_end, test_start, test_end)

    h = talos.Scan(x=[X_train_Scaled_LSTM, X_train_Scaled_ffnn], y=Y_train_Scaled, x_val=[X_train_Scaled_LSTM, X_train_Scaled_ffnn], y_val=Y_train_Scaled, params=pp, model=create_model_MH_DNN,
                   experiment_name='MH_DNN_1-3', round_limit=30, print_params=True)
    h.data.sort_values(by='val_mean_absolute_error', ascending=True)

# Example usage
# perform_hyperparameter_search_MH_DNN(dat, '06/1/2020', '06/6/2021', '06/1/2020', '06/6/2021')


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Support Vector Regression (SVR)
def hyperParameterTuning_SVR(X_train, Y_train):
    param_tuning = {
        'C': [0.01, 0.1],
        'gamma': [0.0001, 0.001, 0.005],
        'epsilon': [0.001, 0.1, 0.3],
        'kernel': ["rbf"]
    }

    svr_regressor = SVR()

    gsearch = GridSearchCV(estimator=svr_regressor,
                           param_grid=param_tuning,
                           scoring='neg_mean_absolute_error',
                           cv=2,
                           n_jobs=-1,
                           verbose=0)

    gsearch.fit(X_train, Y_train)
    return gsearch.best_params_

# XGBoost
def hyperParameterTuning_XGB(X_train, Y_train):
    param_tuning = {
        'max_depth': [1, 2, 4],
        'learning_rate': [.01, 0.03, 0.05, 0.08, 0.1],
        'n_estimators': [25, 50, 75, 100, 200]
    }

    xgb_regressor = XGBRegressor()

    gsearch = GridSearchCV(estimator=xgb_regressor,
                           param_grid=param_tuning,
                           scoring='neg_mean_absolute_error',
                           cv=10,
                           n_jobs=-1,
                           verbose=0)

    gsearch.fit(X_train, Y_train)
    return gsearch.best_params_

# Random Forest
def hyperParameterTuning_RF(X_train, Y_train):
    param_tuning = {
        'bootstrap': [True],
        'max_depth': [40, 60, 80],
        'max_features': [40, 60, 80],
        'min_samples_leaf': [20, 40, 60],
        'min_samples_split': [2, 4, 6],
        'n_estimators': [800, 1600, 2400]
    }

    rf_regressor = RandomForestRegressor()

    gsearch = GridSearchCV(estimator=rf_regressor,
                           param_grid=param_tuning,
                           scoring='neg_mean_absolute_error',
                           cv=10,
                           n_jobs=-1,
                           verbose=3)

    gsearch.fit(X_train, Y_train)
    return gsearch.best_params_

# Example usage
# best_params_svr = hyperParameterTuning_SVR(X_train, Y_train)
# best_params_xgb = hyperParameterTuning_XGB(X_train, Y_train)
# best_params_rf = hyperParameterTuning_RF(X_train, Y_train)


