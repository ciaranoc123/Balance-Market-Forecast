import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta as td
from datetime import datetime
# from pandas import Timedelta as td
import traceback
# from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn_quantile import RandomForestQuantileRegressor
import warnings
from epftoolbox.evaluation import sMAPE
import pandas as pd
import datetime as dt

def load_data_SVR_XGB_RF(file_path):
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



def calculate_metrics_SVR_XGB_RF(data_path):
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

        date_format="%m/%d/%Y %H:%M"

        train_df = None
        

        
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)    
        train_df = participant_df[(participant_df.index>=train_start_time_str) & (participant_df.index<train_end_time_str)].copy(deep="True")

        if train_df is None or len(train_df) == 0:
            print("Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")            
            return train_X, train_y, test_X, test_y, test_df


        
        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)    
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format) 
        test_df = participant_df[(participant_df.index>=test_start_time_str) & (participant_df.index<test_end_time_str)].copy(deep="True")

        if test_df is None or len(test_df) == 0:
            print("Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")            
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
        
#         model.fit(X_train, Y_train) if len(targets) > 1 else model.fit(X_train, Y_train.values.ravel())  
        model.fit(X_train, Y_train) 

        model_test_predictions=None  
        model_test_predictions = model.predict(X_test).reshape(1,16)       
        model_test_predictions=pd.DataFrame(model_test_predictions)
        cols = Y_train.columns.values.tolist()   
        


        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_50"] = model_test_predictions.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions.tolist() 

            
        #print("train number of observations: " + str(len(Y_train)))
        
        
        
           
        return actuals_and_forecast_df
    
    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()
    
def rolling_walk_forward_validation_SVR_XGB_RF(model, data, targets, start_time, end_time, training_days, path):
 
    try:

        all_columns = list(data.columns)            
        results = pd.DataFrame()
            

        date_format="%m/%d/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)
        
        while start_time < end_time:
            
            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time 
    
            test_start_time = train_end_time + td(hours=8)
            test_end_time = test_start_time + td(minutes=30)
            
            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))
    
            train_X, train_y, test_X, test_y, test_df = generate_train_and_test_dataframes_SVR_XGB_RF(participant_df=data, train_start_time=train_start_time.strftime("%m/%d/%Y %H:%M"), train_end_time=train_end_time.strftime("%m/%d/%Y %H:%M"), 
                            test_start_time=test_start_time.strftime("%m/%d/%Y %H:%M"), test_end_time=test_end_time.strftime("%m/%d/%Y %H:%M"))
            
            if train_X is None or len(train_X) == 0:
                print("Don't have a train dataframe for train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
    
            if test_X is None or len(test_X) == 0:
                print("Don't have a test dataframe for test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
            
            actuals_and_forecast_df = fit_multitarget_model_SVR_XGB_RF(model=model, X_train=train_X, Y_train=train_y,
                                            X_test=test_X, Y_test=test_y, actuals_and_forecast_df=test_df,targets=test_df.iloc[:,0:16].columns.values.tolist())
    
            results = pd.concat([results, actuals_and_forecast_df])
            #print(results)
            start_time = start_time + td(hours=8)
            
        results.to_csv(path  + ".csv", index = False)
        
          
        
    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()

