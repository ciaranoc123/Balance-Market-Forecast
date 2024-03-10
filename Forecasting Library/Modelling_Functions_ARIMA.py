
import pandas as pd
import numpy as np
import datetime as dt
from math import floor
import seaborn as sns
from math import sqrt
from datetime import timedelta as td
from statsmodels.tsa.arima.model import ARIMA
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from epftoolbox.evaluation import sMAPE
from sklearn.preprocessing import MinMaxScaler
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn import preprocessing as prep
import warnings





def calculate_and_print_errors_ARIMA(file_path, date_format="%m/%d/%Y %H:%M"):
    """
    Load data from a CSV file, calculate errors, and print them cleanly.

    Parameters:
    - file_path (str): Path to the CSV file containing the results.
    - date_format (str, optional): Date format for parsing. Defaults to "%m/%d/%Y %H:%M".
    """

    # Load the saved file
    results_df = pd.read_csv(file_path)

    # Drop NaN values from each column separately
    Y_test_cleaned = results_df['Actual'].dropna()
    pred_nn_cleaned = results_df['Forecast'].dropna()

    # Function to calculate sMAPE
    def sMAPE(y_true, y_pred):
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

    def calculate_errors(Y_test, pred_nn):
        mse = mean_squared_error(Y_test, pred_nn)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_test, pred_nn)
        smape = sMAPE(Y_test, pred_nn) * 100
        return pd.DataFrame([rmse, mae, smape]).T

    # Calculate errors
    error_metrics = calculate_errors(Y_test_cleaned, pred_nn_cleaned)

    # Rename the columns for clarity
    error_metrics.columns = ['RMSE', 'MAE', 'sMAPE']

    # Print the error metrics
    print("Error Metrics:")
    print(error_metrics)





import pandas as pd
from datetime import datetime as dt

def load_data_ARIMA(file_path, index_col="SettlementPeriod", date_format="%m/%d/%Y %H:%M"):
    """
    Load and preprocess data for ARIMA modeling.

    Parameters:
    - file_path (str): Path to the CSV file.
    - index_col (str, optional): Name of the column to be used as index. Defaults to "SettlementPeriod".
    - date_format (str, optional): Date format for parsing. Defaults to "%m/%d/%Y %H:%M".

    Returns:
    - dat (pd.DataFrame): Processed DataFrame.
    - targets (list): List of target column names.
    """
    # Function to parse dates
    date_parse = lambda date: dt.strptime(date, date_format)

    # Read CSV, parse dates, and set index
    dat = pd.read_csv(file_path, index_col=index_col, parse_dates=True, date_parser=date_parse)

    # Drop unnecessary columns
    dat = dat.drop(["index"], axis=1)

    # Fill missing values
    dat = dat.fillna(method='ffill').fillna(method='bfill')

    # Select only numeric columns
    dat = dat.select_dtypes(include='number')

    # Specify targets for forecasting
    targets = dat.columns[:1]

    return dat, targets










class ARIMAModel(BaseEstimator):
    def __init__(self, order=(1, 0, 0)):
        self.order = order
        self.model = None  # Initialize model attribute

    def fit(self, X, y):
        self.model = ARIMA(y, order=self.order)
        self.model = self.model.fit()
        
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.forecast(steps=len(X))[0]

    def get_params(self, deep=True):
        return {'order': self.order}

    def set_params(self, **params):
        self.order = params['order']
        return self

def rolling_walk_forward_validation_ARIMA(model_fn, data, start_time, end_time, training_days, path, targets):
    try:
        results = pd.DataFrame()
        start_time = dt.strptime(start_time, "%m/%d/%Y %H:%M")
        end_time = dt.strptime(end_time, "%m/%d/%Y %H:%M")

        while start_time < end_time:
            train_start_time = start_time - td(days=training_days)
            train_end_time = start_time

            test_start_time = train_end_time + td(hours=8)
            test_end_time = test_start_time + td(hours=8)  # Forecasting next 8 hours/16 timestamps
            
            # Slice the data for training and testing
            train_X = data.iloc[:, 16:][(data.index >= train_start_time) & (data.index < train_end_time)]
            test_X = data.iloc[:, 16:][(data.index >= test_start_time) & (data.index < test_end_time)]
            
            if train_X.empty or test_X.empty:
                start_time += td(hours=8)  # Move to the next 8-hour period
                continue
                
            train_Y = train_X.iloc[:, :1].values.ravel()  # Select only the first column and convert to 1D array
            test_Y = test_X.iloc[:, :1]
            
            model = model_fn()
            model.fit(train_Y, train_X.iloc[:, 0])  # Ensure train_Y is the target variable
            model_test_predictions = model.predict(test_X)  # Predict for the test set
            actuals_and_forecast_df = pd.DataFrame({'Actual': test_Y.values.ravel(),
                                                     'Forecast': model_test_predictions})
            results = pd.concat([results, actuals_and_forecast_df])
            start_time = test_end_time  # Move to the next forecast period
        results.to_csv(path + ".csv", index=False)

    except Exception as e:
        print("Error:", e)
        

