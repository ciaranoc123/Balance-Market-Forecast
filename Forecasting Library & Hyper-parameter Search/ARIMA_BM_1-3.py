import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta as td
from datetime import datetime, timedelta
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn_quantile import RandomForestQuantileRegressor
from statsmodels.tsa.arima.model import ARIMA
import traceback
from statsmodels.tsa.arima.model import ARIMA
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

# Define date format for parsing
date_format = "%m/%d/%Y %H:%M"

# Function to parse dates
date_parse = lambda date: dt.datetime.strptime(date, date_format)

# Read CSV with frequency information and parse dates
dat = pd.read_csv("/home/ciaran/Documents/BM_data.csv", index_col="SettlementPeriod", parse_dates=True,
                  date_parser=date_parse)

# Specify targets for forecasting
dat = dat.drop(["index"], axis=1)
dat = pd.DataFrame(dat)
dat = dat.bfill(axis='rows')
dat = dat.ffill(axis='rows')
dat = dat._get_numeric_data()
targets = dat.columns[:16]

Y = dat.iloc[:, 0:16]


# Define ARIMA model class
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


# Function to generate train and test dataframes
def generate_train_and_test_dataframes(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
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


def rolling_walk_forward_validation(model_fn, data, start_time, end_time, training_days, path, targets):
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


# Run rolling walk forward validation
rolling_walk_forward_validation(model_fn=fit_arima_model,
                                data=dat, start_time='06/01/2020 00:00', end_time='06/01/2021 00:00',
                                targets=targets, training_days=365,
                                path="/home/ciaran/Documents/BM_results_ARIMAv365_1-12")


