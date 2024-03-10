# Balance-Market-Forecast

This repository contains code and datasets used for forecasting in both the Day-Ahead (DAM) market and the balancing market (BM).

## Purpose
The purpose of this project is to analyze and compare different forecasting models for both the DAM and BM markets, aiming to improve price forecasting accuracy.

## Data Access
Access to the full Day-Ahead market dataset and balancing market dataset via Google Drive link has been removed temporarily while the paper is under review. However, subsets of both datasets, reduced to 25mb in size as per GitHub's restriction, are available in the 'Datasets' folder.

## Code

### Forecasting Library

The 'Forecasting Library' directory contains the main code files used for forecasting. Below is a list of files included:

- BM Forecasting.ipynb: This is the main notebook file containing code for running different forecasting models.
- Modelling_Functions_ARIMA.py: Python script containing functions related to ARIMA modeling.
- Modelling_Functions_LEAR.py: Python script containing functions related to Linear Regression modeling.
- Modelling_Functions_MH_DNN.py: Python script containing functions related to Multi-Head Deep Neural Network modeling.
- Modelling_Functions_SH_DNN.py: Python script containing functions related to Single-Head Deep Neural Network modeling.
- Modelling_Functions_SVR_XGB_RF.py: Python script containing functions related to Support Vector Regression, XGBoost, and Random Forest modeling.

The 'BM Forecasting.ipynb' notebook serves as the main file for running different forecasting models. It imports and utilizes the functions defined in the Python scripts mentioned above.


Please note that access to the full datasets and additional code may be provided upon request or after the paper's review process is complete.

