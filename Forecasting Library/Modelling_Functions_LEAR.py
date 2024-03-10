
from sklearn.metrics import mean_squared_error, mean_absolute_error
from epftoolbox.evaluation import sMAPE

import os;
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from datetime import timedelta as td
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from pandas import concat
import matplotlib.pyplot as plt
from functools import reduce
import importlib
import datetime as dt
from datetime import datetime
from math import floor
import seaborn as sns
from math import sqrt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import LassoLarsIC, Lasso
import warnings

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
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, mean_absolute_error
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

        
        
        
        
        
        
        
        
        
        
        
        
        

def load_and_preprocess_data_LEAR(file_path, date_format="%m/%d/%Y %H:%M"):
    """
    Load data from a CSV file, preprocess it, and return dat, alpha, and Y.

    Parameters:
    - file_path (str): Path to the CSV file.
    - date_format (str, optional): Date format for parsing. Defaults to "%m/%d/%Y %H:%M".

    Returns:
    - dat (pd.DataFrame): Processed DataFrame.
    - alpha (float): Alpha value from LassoLarsIC.
    - Y (pd.DataFrame): Target DataFrame.
    """

    # Load data from CSV
    date_parse = lambda date: dt.strptime(date, date_format)
    dat = pd.read_csv(file_path, index_col="SettlementPeriod", parse_dates=True, date_parser=date_parse)

    # Preprocess data
    dat = dat.drop(["index"], axis=1)
    dat = dat.fillna(method='bfill').fillna(method='ffill')
    dat = dat.select_dtypes(include='number')

    # Define Y
    Y = dat.iloc[:, 0:16]

    # Define X
    X = dat.iloc[:, 16:]

    # Define targets for scaling
    rnn_train_columns = [f'lag_{i}x{j}' for i in range(-3, 18) for j in range(1, 13)]
    rnn_Y_columns = [f'lag_{i}y' for i in range(2, 18)]

    # Scale features
    X_scalers = [MinMaxScaler().fit(X.loc[:, f'lag_{i}x{j}': f'lag_{i-49}x{j}']) for i in range(-3, 18) for j in range(1, 13)]
    rnn_scaled_train = [scaler.transform(X.loc[:, f'lag_{i}x{j}': f'lag_{i-49}x{j}']) for i in range(-3, 18) for j, scaler in enumerate(X_scalers, start=1)]
    X_train_Scaled = np.concatenate(rnn_scaled_train, axis=1)

    # Scale target
    Y_scaler = MinMaxScaler()
    Y_train_Scaled = Y_scaler.fit_transform(Y)

    # Fit LassoLarsIC to get alpha
    alpha = LassoLarsIC(criterion='aic', max_iter=2500).fit(X_train_Scaled, Y_train_Scaled[:, :1].ravel()).alpha_

    return dat, alpha, Y




def process_data_and_get_parameters_LEAR(data_path):
    # Load data
    date_format = "%m/%d/%Y %H:%M"
    date_parse = lambda date: dt.datetime.strptime(date, date_format)
    dat = pd.read_csv(data_path, index_col="SettlementPeriod", parse_dates=True, date_parser=date_parse)
    dat = dat.drop(["index"], axis=1)
    dat = dat.bfill(axis ='rows')
    dat = dat.ffill(axis ='rows')
    dat = dat._get_numeric_data()

    # Split data into Y and X
    Y = dat.iloc[:, 0:16]
    X = dat.iloc[:, 16:]

    # Train-test split
    X_train = X.iloc[:7250,:]
    Y_train = Y.iloc[:7250,:]
    X_test = X.iloc[7250:8739,:]
    Y_test = Y.iloc[7250:8739,:]

    # Define subsets for RNN training
    rnn_train_1 = X_train.loc[:, "lag_-3x1": "lag_-50x1"]
    rnn_train_2 = X_train.loc[:, "lag_-3x2": "lag_-50x2"]
    rnn_train_3 = X_train.loc[:, "lag_-2x3": "lag_-49x3"]
    rnn_train_4 = X_train.loc[:, "lag_0x6": "lag_-47x6"]
    rnn_train_5 = X_train.loc[:, "lag_-2x12": "lag_-49x12"]
    rnn_train_6 = X_train.loc[:, "lag_2x7": "lag_17x7"]
    rnn_train_7 = X_train.loc[:, "lag_2x8": "lag_17x8"]
    rnn_train_8 = X_train.loc[:, "lag_2x9": "lag_17x9"]
    rnn_train_9 = X_train.loc[:, "lag_2x10": "lag_17x10"]
    rnn_train_10 = X_train.loc[:, "lag_2x11": "lag_17x11"]

    # Subset for Y (target)
    rnn_Y = Y_train.loc[:, "lag_2y":"lag_17y"]

    # Scale features and target
    X_scalers = []
    rnn_scaled_train = []
    for rnn_train in [rnn_train_1, rnn_train_2, rnn_train_3, rnn_train_4, rnn_train_5,
                      rnn_train_6, rnn_train_7, rnn_train_8, rnn_train_9, rnn_train_10]:
        X_scaler = preprocessing.MinMaxScaler()
        rnn_scaled = X_scaler.fit_transform(rnn_train)
        X_scalers.append(X_scaler)
        rnn_scaled_train.append(rnn_scaled)

    Y_scaler = preprocessing.MinMaxScaler()
    Y_train_Scaled = Y_scaler.fit_transform(rnn_Y)

    X_train_Scaled = np.concatenate(rnn_scaled_train, axis=1)

    # LassoLarsIC parameter
    alpha = LassoLarsIC(criterion='aic', max_iter=2500, normalize=False).fit(X_train_Scaled, Y_train_Scaled[:,:1].ravel()).alpha_

    return Y, alpha, dat




def calculate_metrics_LEAR(data_path):
    # Load data
    date_format = "%m/%d/%Y %H:%M"
    date_parse = lambda date: dt.datetime.strptime(date, date_format)
    dat1 = pd.read_csv(data_path)
    dat1 = pd.DataFrame(dat1)

    # Ignore DeprecationWarning
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Extract necessary columns
    column_names = ['lag_{}y_Forecast_50'.format(i) for i in range(2, 18)]
    Q_50 = dat1[column_names].dropna().stack().reset_index()
    column_names = ['lag_{}y'.format(i) for i in range(2, 18)]
    YY_test = dat1[column_names].dropna().stack().reset_index()

    # Step 1: Calculate MAE
    mae = mean_absolute_error(YY_test[0], Q_50[0])

    # Step 2: Calculate RMSE
    mse = mean_squared_error(YY_test[0], Q_50[0])
    rmse = np.sqrt(mse)

    # Step 3: Calculate sMAPE
    smape = sMAPE(YY_test[0], Q_50[0]) * 100

    # Print metrics
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Symmetric Mean Absolute Percentage Error (sMAPE):", smape)

    # Reset warnings to default behavior
    warnings.resetwarnings()

    return mae, rmse, smape

# Example usage:


def generate_train_and_test_dataframes_LEAR(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                       test_start_time: dt, test_end_time: dt):


    # These are the dataframes that will be returned from the method.
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


        rnn_train_1 = train_df.loc[:,"lag_-3x1": "lag_-50x1"]
        rnn_train_2 = train_df.loc[:,"lag_-3x2":"lag_-50x2"]
        rnn_train_3 = train_df.loc[:,"lag_-2x3": "lag_-49x3"]
        rnn_train_4 = train_df.loc[:,"lag_0x6": "lag_-47x6"]
        rnn_train_5 = train_df.loc[:,"lag_-2x12": "lag_-49x12"]
        rnn_train_6 = train_df.loc[:,"lag_2x7":"lag_17x7"]
        rnn_train_7 = train_df.loc[:,"lag_2x8": "lag_17x8"]
        rnn_train_8 = train_df.loc[:,"lag_2x9":"lag_17x9"]
        rnn_train_9 = train_df.loc[:,"lag_2x10":"lag_17x10"]
        rnn_train_10 = train_df.loc[:,"lag_2x11": "lag_17x11"]

        rnn_test_1 = test_df.loc[:,"lag_-3x1": "lag_-50x1"]
        rnn_test_2 = test_df.loc[:,"lag_-3x2":"lag_-50x2"]
        rnn_test_3 = test_df.loc[:,"lag_-2x3":"lag_-49x3"]
        rnn_test_4 = test_df.loc[:,"lag_0x6":"lag_-47x6"]
        rnn_test_5 = test_df.loc[:,"lag_-2x12":"lag_-49x12"]
        rnn_test_6 = test_df.loc[:,"lag_2x7":"lag_17x7"]
        rnn_test_7 = test_df.loc[:,"lag_2x8":"lag_17x8"]
        rnn_test_8 = test_df.loc[:,"lag_2x9":"lag_17x9"]
        rnn_test_9 = test_df.loc[:,"lag_2x10":"lag_17x10"]
        rnn_test_10 = test_df.loc[:,"lag_2x11":"lag_17x11"]

        rnn_Y = train_df.loc[:,"lag_2y":"lag_17y"]

        X_scaler_1 = preprocessing.MinMaxScaler()
        X_scaler_2 = preprocessing.MinMaxScaler()
        X_scaler_3 = preprocessing.MinMaxScaler()
        X_scaler_4 = preprocessing.MinMaxScaler()
        X_scaler_5 = preprocessing.MinMaxScaler()
        X_scaler_6 = preprocessing.MinMaxScaler()
        X_scaler_7 = preprocessing.MinMaxScaler()
        X_scaler_8 = preprocessing.MinMaxScaler()
        X_scaler_9 = preprocessing.MinMaxScaler()
        X_scaler_10 = preprocessing.MinMaxScaler()

        Y_scaler = preprocessing.MinMaxScaler()

        rnn_scaled_train_1 = X_scaler_1.fit_transform(rnn_train_1)
        rnn_scaled_train_2 = X_scaler_2.fit_transform(rnn_train_2)
        rnn_scaled_train_3 = X_scaler_3.fit_transform(rnn_train_3)
        rnn_scaled_train_4 = X_scaler_4.fit_transform(rnn_train_4)
        rnn_scaled_train_5 = X_scaler_5.fit_transform(rnn_train_5)
        rnn_scaled_train_6 = X_scaler_6.fit_transform(rnn_train_6)
        rnn_scaled_train_7 = X_scaler_7.fit_transform(rnn_train_7)
        rnn_scaled_train_8 = X_scaler_8.fit_transform(rnn_train_8)
        rnn_scaled_train_9 = X_scaler_9.fit_transform(rnn_train_9)
        rnn_scaled_train_10 = X_scaler_10.fit_transform(rnn_train_10)

        train_y = Y_scaler.fit_transform(rnn_Y)
        Y_scaler_n=Y_scaler.fit(rnn_Y)
        
        train_X=np.concatenate((rnn_scaled_train_1, rnn_scaled_train_2, rnn_scaled_train_3, rnn_scaled_train_4, rnn_scaled_train_5,
                                rnn_scaled_train_6, rnn_scaled_train_7, rnn_scaled_train_8, rnn_scaled_train_9, rnn_scaled_train_10), axis=1)
        test_X=np.concatenate((X_scaler_1.transform(rnn_test_1), X_scaler_2.transform(rnn_test_2), X_scaler_3.transform(rnn_test_3), X_scaler_4.transform(rnn_test_4), X_scaler_5.transform(rnn_test_5), 
                               X_scaler_6.transform(rnn_test_6), X_scaler_7.transform(rnn_test_7), X_scaler_8.transform(rnn_test_8),X_scaler_9.transform(rnn_test_9), X_scaler_10.transform(rnn_test_10)), axis=1)


        test_y = test_df.iloc[:, 0:16]

        return train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n


def fit_multitarget_model_LEAR(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets, Y_scaler_n):
    try:
        Y_scaler = preprocessing.MinMaxScaler()
        Y_scaler = Y_scaler.fit(Y_train)
        cols = Y_test.iloc[:, 0:16].columns.values.tolist()

        model.fit(X_train, Y_train) 


        model_test_predictions=None  
  
        model_test_predictions = pd.DataFrame(Y_scaler_n.inverse_transform(np.array(model.predict(X_test)).reshape(1,16)), columns=cols)    

        #print(model_test_predictions)

        
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_50"] = model_test_predictions.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions.tolist() 
            

        return actuals_and_forecast_df

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()


  
    
def rolling_walk_forward_validation_LEAR(model, data, targets, start_time, end_time, training_days, path):
 
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
            train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n= generate_train_and_test_dataframes_LEAR(
                participant_df=data, train_start_time=train_start_time.strftime("%m/%d/%Y %H:%M"), train_end_time=train_end_time.strftime("%m/%d/%Y %H:%M"), 
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
            actuals_and_forecast_df = fit_multitarget_model_LEAR(model=model,
                                                            Y_scaler_n=Y_scaler_n, X_train=train_X, Y_train=train_y, X_test=test_X, Y_test=test_y,
                                                            actuals_and_forecast_df=test_df.iloc[:,0:16], targets=test_df.iloc[:,0:16].columns.values.tolist())

    
            results = results.append(actuals_and_forecast_df)
        
            start_time = start_time + td(hours=8)
            
        results.to_csv(path  + ".csv", index = False)
        

        
    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()




































        
        
        
 









