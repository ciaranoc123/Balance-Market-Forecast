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






def load_data_SH_DNN(file_path):
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



def model_SH_DNN():
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
    
    i_shape = (16, 20)
    net_input = Input(shape=i_shape)    

    x = Flatten()(net_input)
    
    for _ in range(2):
        x = Dense(192, activation='sigmoid')(x)
    
    for _ in range(3):
        x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    c = Dense(64, activation='sigmoid')(x)
    
    for _ in range(1):
        c = Dense(64, activation='relu')(c) 
    output_3 = Dense(16, name='out_50')(c)
    
    opt = Adam(learning_rate=0.0001)
    model = Model(inputs=net_input, outputs=output_3)    
    model.compile(loss=loss_50th_p, optimizer=opt)
    
    return model



mmo = KerasRegressor(build_fn=model_SH_DNN, epochs=20, batch_size=16, verbose=2)


def calculate_metrics_SH_DNN(results):
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
#             return train_X, train_y, test_X, test_y, test_df
        
        original_columns = list(participant_df.columns)

        #Remove any rows with nan's etc (there shouldn't be any in the input).        
        participant_df = participant_df.dropna()
        
        date_format="%m/%d/%Y %H:%M"
      
        train_df = None
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
        train_df = participant_df[(participant_df.index>=train_start_time_str) & (participant_df.index<train_end_time_str)].copy(deep="True")

        if train_df is None or len(train_df) == 0:
            print("Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")            

        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)    
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format) 
        test_df = participant_df[(participant_df.index>=test_start_time_str) & (participant_df.index<test_end_time_str)].copy(deep="True")

        if test_df is None or len(test_df) == 0:
            print("Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")            

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

        rnn_Y = train_df.loc[:, "lag_2y":"lag_17y"]


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

        train_y   = Y_scaler.fit_transform(rnn_Y)
        Y_scaler_n=Y_scaler.fit(rnn_Y)
        train_X = np.hstack(
            (rnn_scaled_train1_a, rnn_scaled_train1_b, rnn_scaled_train1_c, rnn_scaled_train2_a, rnn_scaled_train2_b, rnn_scaled_train2_c,
             rnn_scaled_train3_a, rnn_scaled_train3_b, rnn_scaled_train3_c, rnn_scaled_train4_a, rnn_scaled_train4_b, rnn_scaled_train4_c,
             rnn_scaled_train5_a, rnn_scaled_train5_b, rnn_scaled_train5_c,rnn_scaled_train6,rnn_scaled_train7, rnn_scaled_train8,
             rnn_scaled_train9, rnn_scaled_train10)
        ).reshape(rnn_train6.shape[0], 20, 16).transpose(0, 2, 1)

        test_X = np.hstack(
            (X_scaler1_a.transform(rnn_test1_a),X_scaler1_b.transform(rnn_test1_b),X_scaler1_c.transform(rnn_test1_c),
             X_scaler2_a.transform(rnn_test2_a),X_scaler2_b.transform(rnn_test2_b),X_scaler2_c.transform(rnn_test2_c),
             X_scaler3_a.transform(rnn_test3_a),X_scaler3_b.transform(rnn_test3_b),X_scaler3_c.transform(rnn_test3_c),
             X_scaler4_a.transform(rnn_test4_a),X_scaler4_b.transform(rnn_test4_b),X_scaler4_c.transform(rnn_test4_c),
             X_scaler5_a.transform(rnn_test5_a),X_scaler5_b.transform(rnn_test5_b),X_scaler5_c.transform(rnn_test5_c),
             X_scaler6.transform(rnn_test6),X_scaler7.transform(rnn_test7),X_scaler8.transform(rnn_test8),
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
        cols = pd.DataFrame(Y_train).columns.values.tolist()                         
        es = EarlyStopping(monitor='val_loss', mode='min',  patience=30)
        i_shape=(X_train.shape[1], X_train.shape[2])

        model.fit(X_train, Y_train, epochs=300, verbose=0,  callbacks=[es], validation_split=0.1) 
        model_test_predictions=None        

        model_test_predictions = pd.DataFrame(Y_scaler_n.inverse_transform(model.predict(X_test).reshape(1,16)), columns=cols)
        #print(model_test_predictions)


        

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[str(cols[i]) + "_Forecast_50"] = model_test_predictions.iloc[:, i].T.tolist() if len(cols) > 1 else model_test_predictions.tolist()

            #actuals_and_forecast_df[cols[i]+"_Forecast_50"] = model_test_predictions.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions.tolist()             

            
        return actuals_and_forecast_df
    
    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()
  
    
def rolling_walk_forward_validation_SH_DNN(model, data, targets, start_time, end_time, training_days, path):
 
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
    
            #Generate the calibration and test dataframes.
            train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n= generate_train_and_test_dataframes_SH_DNN(participant_df=data, train_start_time=train_start_time.strftime("%m/%d/%Y %H:%M"), train_end_time=train_end_time.strftime("%m/%d/%Y %H:%M"), 
                            test_start_time=test_start_time.strftime("%m/%d/%Y %H:%M"), test_end_time=test_end_time.strftime("%m/%d/%Y %H:%M"))
            
            
            if train_X is None or len(train_X) == 0:
                print("Don't have a train dataframe for train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
    
            if test_X is None or len(test_X) == 0:
                print("Don't have a test dataframe for test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
            
            #Fit the model to the train datasets, produce a forecast and return a dataframe containing the forecast/actuals.
            actuals_and_forecast_df = fit_multitarget_model_SH_DNN(model=model,Y_scaler_n=Y_scaler_n, X_train=train_X, Y_train=train_y, 
                                            X_test=test_X, Y_test=test_y, actuals_and_forecast_df=test_df.iloc[:,0:16], targets=test_df.iloc[:,0:16].columns.values.tolist())

    
            results = results.append(actuals_and_forecast_df)
        
            start_time = start_time + td(hours=8)
            
        results.to_csv(path  + ".csv", index = False)
        

        
    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()




