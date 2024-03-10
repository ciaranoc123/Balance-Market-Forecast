import datetime as dt 
from datetime import datetime
from datetime import timedelta as td
import traceback
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pandas import concat
from functools import reduce
import importlib
import datetime as dt
from datetime import datetime
from math import floor
import seaborn as sns
from math import sqrt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.layers import concatenate
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K
from epftoolbox.evaluation import sMAPE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn import preprocessing as prep
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import make_scorer, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from hyperopt import hp, fmin, tpe
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)


def load_data_MH_DNN(file_path):
    # Define date parsing function
    date_format = "%m/%d/%Y %H:%M"
    date_parse = lambda date: dt.datetime.strptime(date, date_format)
    
    # Load the data
    dat = pd.read_csv(file_path, index_col="SettlementPeriod", parse_dates=True, date_parser=date_parse)
    
    # Drop the 'index' column if present
    if 'index' in dat.columns:
        dat = dat.drop(columns=['index'])
    
    # Perform backward and forward filling
    dat = dat.bfill(axis='rows').ffill(axis='rows')
    
    # Keep only numeric columns
    dat = dat.select_dtypes(include='number')
    
    return dat



def qloss(qs, y_true, y_pred):
    # Pinball loss for multiple quantiles
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

loss_10th_p = lambda y_true, y_pred: qloss(0.1, y_true, y_pred)
loss_30th_p = lambda y_true, y_pred: qloss(0.3, y_true, y_pred)
loss_50th_p = lambda y_true, y_pred: qloss(0.5, y_true, y_pred)
loss_70th_p = lambda y_true, y_pred: qloss(0.7, y_true, y_pred)
loss_90th_p = lambda y_true, y_pred: qloss(0.9, y_true, y_pred)



def create_multiheaded_model(input_shape_lstm, input_shape_ffnn):
    visible1 = layers.Input(shape=input_shape_lstm, name='input_lstm')
    x1 = visible1
    for i in range(1):
        x1 = layers.LSTM(64, return_sequences=True, activation='sigmoid')(x1)
    x1 = layers.Dropout(0.222222)(x1)
    input_1 = layers.Flatten()(x1)

    visible2 = layers.Input(shape=input_shape_ffnn, name='input_ffnn')
    x2 = visible2
    for i in range(2):
        x2 = layers.Dense(64, activation='tanh')(x2)
    x2 = layers.Dropout(0.088889)(x2)
    input_2 = layers.Flatten()(x2)

    merged = layers.Concatenate()([input_1, input_2])
    x = layers.Dense(128, activation='tanh')(merged)
    
    x = layers.Dense(128, activation='relu')(x)
    output_1 = layers.Dense(16, name='out_50')(x)

    model = tf.keras.Model(inputs=[visible1, visible2], outputs=output_1)
    opt = tf.keras.optimizers.Adam(learning_rate=0.000100)

    model.compile(loss=loss_50th_p, optimizer=opt)
    return model
mmo = KerasRegressor(build_fn=create_multiheaded_model, epochs=20, batch_size=32, verbose=2)
    
    
def calculate_metrics_MH_DNN(results):
    # Extract necessary columns
    column_names = ['{}_Forecast_50'.format(i) for i in range(0, 16)]

    Q_50 = results[column_names].dropna().stack().reset_index(drop=True)
    column_names = ['lag_{}y'.format(i) for i in range(2, 18)]
    YY_test = results[column_names].dropna().stack().reset_index(drop=True)

    # Step 1: Calculate MAE
    mae = mean_absolute_error(YY_test, Q_50)

    # Step 2: Calculate RMSE
    mse = mean_squared_error(YY_test, Q_50)
    rmse = np.sqrt(mse)

    # Step 3: Calculate sMAPE
    smape = sMAPE(YY_test, Q_50) * 100

    # Print metrics
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Symmetric Mean Absolute Percentage Error (sMAPE):", smape)

    return mae, rmse, smape
    
def generate_train_and_test_dataframes_MH_RNN_DNN(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                       test_start_time: dt, test_end_time: dt):

    # These are the dataframes that will be returned from the method.
    train_X_LSTM = None
    train_X_ffnn = None
    train_y = None
    test_X_LSTM = None
    test_X_ffnn = None
    test_y = None
    test_df = None
    train_df = None
    Y_scaler_n = None

    try:
        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
        
        original_columns = list(participant_df.columns)

        # Remove any rows with NaNs etc (there shouldn't be any in the input).
        participant_df = participant_df.dropna()

        date_format = "%m/%d/%Y %H:%M"
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
        train_df = participant_df[(participant_df.index >= train_start_time_str) & (participant_df.index < train_end_time_str)].copy(deep=True)

        if train_df is None or len(train_df) == 0:
            print(f"Don't have a train dataframe for train_start_time: {train_start_time_str}, train_end_time: {train_end_time_str}, exiting.")            

        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)    
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format) 
        test_df = participant_df[(participant_df.index >= test_start_time_str) & (participant_df.index < test_end_time_str)].copy(deep=True)

        if test_df is None or len(test_df) == 0:
            print(f"Don't have a test dataframe for test_start_time: {test_start_time_str}, test_end_time: {test_end_time_str}, exiting.")            

        rnn_Y = train_df.loc[:, "lag_2y":"lag_17y"]

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

        rnn_scaled_train_LSTM_1 = X_scaler_LSTM_1.fit_transform(train_df.loc[:, "lag_-3x1": "lag_-50x1"])
        rnn_scaled_train_LSTM_2 = X_scaler_LSTM_2.fit_transform(train_df.loc[:, "lag_-3x2":"lag_-50x2"])
        rnn_scaled_train_LSTM_3 = X_scaler_LSTM_3.fit_transform(train_df.loc[:, "lag_-2x3":"lag_-49x3"])
        rnn_scaled_train_LSTM_4 = X_scaler_LSTM_4.fit_transform(train_df.loc[:, "lag_0x6":"lag_-47x6"])
        rnn_scaled_train_LSTM_5 = X_scaler_LSTM_5.fit_transform(train_df.loc[:, "lag_-2x12":"lag_-49x12"])

        rnn_scaled_train_ffnn_1 = X_scaler_ffnn_1.fit_transform(train_df.loc[:, "lag_2x7":"lag_17x7"])
        rnn_scaled_train_ffnn_2 = X_scaler_ffnn_2.fit_transform(train_df.loc[:, "lag_2x8":"lag_17x8"])
        rnn_scaled_train_ffnn_3 = X_scaler_ffnn_3.fit_transform(train_df.loc[:, "lag_2x9":"lag_17x9"])
        rnn_scaled_train_ffnn_4 = X_scaler_ffnn_4.fit_transform(train_df.loc[:, "lag_2x10":"lag_17x10"])
        rnn_scaled_train_ffnn_5 = X_scaler_ffnn_5.fit_transform(train_df.loc[:, "lag_2x11":"lag_17x11"])

        train_y = Y_scaler.fit_transform(rnn_Y)
        Y_scaler_n = Y_scaler.fit(rnn_Y)

        train_X_LSTM = np.hstack(
            (rnn_scaled_train_LSTM_1, rnn_scaled_train_LSTM_2, rnn_scaled_train_LSTM_3,
             rnn_scaled_train_LSTM_4, rnn_scaled_train_LSTM_5)
        ).reshape(train_df.shape[0], 5, 48).transpose(0, 2, 1)

        train_X_ffnn = np.hstack(
            (rnn_scaled_train_ffnn_1, rnn_scaled_train_ffnn_2, rnn_scaled_train_ffnn_3,
             rnn_scaled_train_ffnn_4, rnn_scaled_train_ffnn_5)
        ).reshape(train_df.shape[0], 5, 16).transpose(0, 2, 1)

        test_X_LSTM = np.hstack(
            (X_scaler_LSTM_1.transform(test_df.loc[:, "lag_-3x1":"lag_-50x1"]),
             X_scaler_LSTM_2.transform(test_df.loc[:, "lag_-3x2":"lag_-50x2"]),
             X_scaler_LSTM_3.transform(test_df.loc[:, "lag_-2x3":"lag_-49x3"]),
             X_scaler_LSTM_4.transform(test_df.loc[:, "lag_0x6":"lag_-47x6"]),
             X_scaler_LSTM_5.transform(test_df.loc[:, "lag_-2x12":"lag_-49x12"]))
        ).reshape(test_df.shape[0], 5, 48).transpose(0, 2, 1)

        test_X_ffnn = np.hstack(
            (X_scaler_ffnn_1.transform(test_df.loc[:, "lag_2x7":"lag_17x7"]),
             X_scaler_ffnn_2.transform(test_df.loc[:, "lag_2x8":"lag_17x8"]),
             X_scaler_ffnn_3.transform(test_df.loc[:, "lag_2x9":"lag_17x9"]),
             X_scaler_ffnn_4.transform(test_df.loc[:, "lag_2x10":"lag_17x10"]),
             X_scaler_ffnn_5.transform(test_df.loc[:, "lag_2x11":"lag_17x11"]))
        ).reshape(test_df.shape[0], 5, 16).transpose(0, 2, 1)

        test_y = test_df.iloc[:, 0:16]

        return train_X_LSTM, train_X_ffnn, train_y, test_X_LSTM, test_X_ffnn, test_y, test_df, train_df, Y_scaler_n

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X_LSTM, train_X_ffnn, train_y, test_X_LSTM, test_X_ffnn, test_y, test_df, train_df, Y_scaler_n



def fit_multitarget_model_MH_RNN_DNN(model, X_train_LSTM, X_train_ffnn, Y_train, X_test_LSTM, X_test_ffnn, Y_test, actuals_and_forecast_df, targets, Y_scaler_n):

    try:
        Y_scaler = preprocessing.MinMaxScaler()
        Y_scaler = Y_scaler.fit(Y_train)
        cols = pd.DataFrame(Y_train).columns.values.tolist()   
        
        model = create_multiheaded_model((48, 5), (16, 5))
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        es = EarlyStopping(monitor='val_loss', mode='min',  patience=30)
        model.fit([X_train_LSTM, X_train_ffnn], Y_train, epochs=300, verbose=0,  callbacks=[es], validation_split=0.10)
        
        model_test_predictions = None
        model_test_predictions = pd.DataFrame(Y_scaler.inverse_transform(np.array(model.predict([X_test_LSTM, X_test_ffnn])).reshape(1, 16)), columns=cols)
            

        #print(model_test_predictions)
        #print("test number of observations: " + str(len(Y_test)))

        
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[str(cols[i]) + "_Forecast_50"] = model_test_predictions.iloc[:, i].T.tolist() if len(cols) > 1 else model_test_predictions.tolist()

        return actuals_and_forecast_df

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()





def rolling_walk_forward_validation_MH_RNN_DNN(model, data, targets, start_time, end_time, training_days, path):
 
    try:

        all_columns = list(data.columns)            
        results = pd.DataFrame()
            

        date_format="%m/%d/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)
        
        while start_time < end_time:
            
            #Train interval
            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time
    
            #Test interval, the test period is always the day ahead forecast
            test_start_time = train_end_time + td(hours=8)
            test_end_time = test_start_time + td(minutes=30)
            
            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))

            # Generate the calibration and test dataframes.
            train_X_LSTM, train_X_ffnn, train_y, test_X_LSTM, test_X_ffnn, test_y, test_df, train_df, Y_scaler_n = generate_train_and_test_dataframes_MH_RNN_DNN(
                participant_df=data, train_start_time=train_start_time.strftime("%m/%d/%Y %H:%M"), train_end_time=train_end_time.strftime("%m/%d/%Y %H:%M"), 
                test_start_time=test_start_time.strftime("%m/%d/%Y %H:%M"), test_end_time=test_end_time.strftime("%m/%d/%Y %H:%M"))

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
            actuals_and_forecast_df = fit_multitarget_model_MH_RNN_DNN(model=model, X_train_LSTM=train_X_LSTM, X_train_ffnn=train_X_ffnn,
                                                            Y_train=train_y, X_test_LSTM=test_X_LSTM, X_test_ffnn=test_X_ffnn,
                                                            Y_test=test_y, Y_scaler_n=Y_scaler_n, actuals_and_forecast_df=test_df.iloc[:, 0:16], targets=test_df.iloc[:,0:16].columns.values.tolist())

            results = results.append(actuals_and_forecast_df)
        
            start_time = start_time + td(hours=8)
            
        results.to_csv(path  + ".csv", index = False)
        

        
    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()






























