#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 04:18:11 2022

@author: ciaran
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:02:46 2022

@author: ciaran
"""

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
from sklearn.linear_model import Lasso, LinearRegression


date_format="%m/%d/%Y %H:%M"
date_parse = lambda date: dt.datetime.strptime(date, date_format)
# dat = pd.read_csv("C:/Users/ciara/OneDrive/Documents/BM_data.csv", index_col="SettlementPeriod", parse_dates=True, date_parser=date_parse)
dat = pd.read_csv("/home/coconnor/BM_data.csv", index_col="SettlementPeriod", parse_dates=True, date_parser=date_parse)

# dat = pd.read_csv("/home/ciaran/Documents/BM_data.csv", index_col="SettlementPeriod", parse_dates=True, date_parser=date_parse)

dat = dat.drop(["index"], axis=1)
dat=pd.DataFrame(dat)
dat=dat.bfill(axis ='rows')
dat=dat.ffill(axis ='rows')
dat=dat._get_numeric_data()

Y=dat.iloc[:, 0:16]

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
  
    #These are the dataframes that will be returned from the method.
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

        #Remove any rows with nan's etc (there shouldn't be any in the input).        
        participant_df = participant_df.dropna()

        date_format="%m/%d/%Y %H:%M"

        #The train dataframe, it will be used later to create train_X and train_y.
        train_df = None
        
#         train_start_time_str = train_start_time.strptime(date_format)
#         train_end_time_str = train_end_time.strptime(date_format)
        
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)    
        train_df = participant_df[(participant_df.index>=train_start_time_str) & (participant_df.index<train_end_time_str)].copy(deep="True")

        if train_df is None or len(train_df) == 0:
            print("Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")            
            return train_X, train_y, test_X, test_y, test_df

        #Create the test dataframe, it will be used later to create test_X and test_y
#         test_start_time_str = test_start_time.strftime(date_format)
#         test_end_time_str = test_end_time.strftime(date_format)
        
        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)    
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format) 
        test_df = participant_df[(participant_df.index>=test_start_time_str) & (participant_df.index<test_end_time_str)].copy(deep="True")

        if test_df is None or len(test_df) == 0:
            print("Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")            
            return train_X, train_y, test_X, test_y, test_df

        #The if statement handles situations where we want to scale explanatory variables (need to take care in case we one hot encoded explanatory variables)

        train_X = train_df.iloc[:, 16:]
        test_X = test_df.iloc[:, 16:]
        train_y = train_df.iloc[:, 0:16]
        test_y = test_df.iloc[:, 0:16]
                                
        return train_X, train_y, test_X, test_y, test_df
        
    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df
    
    
def fit_multitarget_model(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets):
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
        
#         model.fit(X_train, Y_train) if len(targets) > 1 else model.fit(X_train, Y_train.values.ravel())  
        model.fit(X_train, Y_train) 

        model_test_predictions=None  
        model_train_predictions=None
        model_train_predictions = model.predict(X_train)
        model_test_predictions = model.predict(X_test)          
                    
        cols = Y_train.columns.values.tolist()   
        
        
        print("train number of observations: " + str(len(Y_train)))
        model_train_mse = mean_squared_error(Y_train, model_train_predictions)
        model_train_rmse = round(np.sqrt(model_train_mse),2)
        model_train_mae = round(mean_absolute_error(Y_train, model_train_predictions),2)
        print("train rmse: " + str(model_train_rmse))
        print("train mae: " + str(model_train_mae))
        
        model_test_mse = mean_squared_error(Y_test, model_test_predictions)
        model_test_rmse = round(np.sqrt(model_test_mse),2)
        model_test_mae = round(mean_absolute_error(Y_test, model_test_predictions),2)
        print("test rmse: " + str(model_test_rmse))
        print("test mae: " + str(model_test_mae))
        
        
        
        for i in range(0,len(cols)):    
            predictor_train_mse = mean_squared_error(Y_train[cols[i]], model_train_predictions[:,i]) if len(cols) > 1 else mean_squared_error(Y_train[cols[i]], model_train_predictions.tolist()) 
            predictor_train_rmse = round(np.sqrt(predictor_train_mse),2)
            predictor_train_mae = round(mean_absolute_error(Y_train[cols[i]], model_train_predictions[:,i]),2) if len(cols) > 1 else round(mean_absolute_error(Y_train[cols[i]], model_train_predictions.tolist()),2)
            print(cols[i] + " train rmse: " + str(predictor_train_rmse))
            print(cols[i] + " train mae: " + str(predictor_train_mae))
            
            
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast"] = model_test_predictions[:,i].tolist() if len(cols) > 1 else model_test_predictions.tolist() 
            predictor_test_mse = mean_squared_error(Y_test[cols[i]], model_test_predictions[:,i]) if len(cols) > 1 else mean_squared_error(Y_test[cols[i]], model_test_predictions.tolist())
            predictor_test_rmse = round(np.sqrt(predictor_test_mse), 2)
            predictor_test_mae = round(mean_absolute_error(Y_test[cols[i]], model_test_predictions[:,i]),2) if len(cols) > 1 else round(mean_absolute_error(Y_test[cols[i]], model_test_predictions.tolist()),2)
            print(cols[i] + " test rmse: " + str(predictor_test_rmse))
            print(cols[i] + " test mae: " + str(predictor_test_mae))
            
        Error_i= ([model_test_rmse, model_test_mae, model_train_rmse, model_train_mae])
        print(Error_i)
        actuals_and_forecast_df = actuals_and_forecast_df.append(Error_i)
          
        #return the test set, the target and forecast values.               
#         actuals_and_forecast_df.sort_index(inplace=True)
#         test_columns = [actuals_and_forecast_df.index]
#         for i in range(0,len(cols)):
#             test_columns.extend([cols[i]+"_Forecast",cols[i]])
#         Error_i=Error_i   
#         actuals_and_forecast_df = actuals_and_forecast_df[test_columns]
#         #return the test set, the target and forecast values.               
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
            
        #Each time we 
        # (a) fit the model on the calibration/train data
        # (b) apply it to the test data i.e. forecast 1 day ahead.
        #Repeat.
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
            train_X, train_y, test_X, test_y, test_df = generate_train_and_test_dataframes(participant_df=dat, train_start_time=train_start_time.strftime("%m/%d/%Y %H:%M"), train_end_time=train_end_time.strftime("%m/%d/%Y %H:%M"), 
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
            actuals_and_forecast_df = fit_multitarget_model(model=model, X_train=train_X, Y_train=train_y,
                                            X_test=test_X, Y_test=test_y, actuals_and_forecast_df=test_df,targets=Y.columns.values.tolist())
    
            results = results.append(actuals_and_forecast_df)
            print(results)
            start_time = start_time + td(minutes=30)
            
        results.to_csv(path  + ".csv", index = False)
        
        #plot the output
#         if results is not None and len(results)  > 0:
#             for i in range(0,len(targets)):    
#                 title_str = scenario + ", " + targets[i] + ": forecast vs. actual"
#                 results.plot(y=[targets[i]+"_Forecast", targets[i]], x=date_time_column,style=['bs-', 'ro-'],title=title_str)            
        
    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()

rolling_walk_forward_validation(model=RandomForestRegressor(n_estimators =1400, max_features = 15, max_depth = 70, bootstrap = True, 
                                                            min_samples_split= 8, min_samples_leaf = 90 ), 
                                data=dat, start_time='12/1/2020 00:00',end_time='3/1/2021  00:00',
                                targets=Y.columns.values.tolist(),training_days=-30, path="/home/coconnor/BM_results_RF_16_30")