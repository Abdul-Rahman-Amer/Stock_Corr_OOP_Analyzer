import time
import concurrent.futures
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
    
def make_all_forecasts(Stock1,Stock2):  
    df_a = pd.DataFrame(Stock1['results'])
    df_b = pd.DataFrame(Stock2['results'])
    df_a[['o','c']] = np.log(df_a[['o','c']]).diff()

    df_b[['o','c']] = np.log(df_b[['o','c']]).diff()

    df_a['t'] = pd.to_datetime(df_a['t'])
    df_b['t'] = pd.to_datetime(df_b['t'])

    df_a_o = df_a[['o']]
    df_b_o = df_b[['o']]
    
    df_a_c = df_a[['c']]
    df_b_c = df_b[['c']]

    df_a_o.fillna(method='bfill', inplace=True)
    df_a_c.fillna(method='bfill', inplace=True)
    df_b_o.fillna(method='bfill', inplace=True)
    df_b_c.fillna(method='bfill', inplace=True)
    

    open_corr = []
    close_corr = []

    a = 0
    b = 252
    len_data=len(df_a)
    while b < len_data:

        c,p = stats.pearsonr(df_a_o[a:b]['o'], df_b_o[a:b]['o'])
        open_v = c
        c,p = stats.pearsonr(df_a_c[a:b]['c'], df_b_c[a:b]['c'])
        close_v = c
        open_corr.append([open_v,df_a['t'][b]])
        close_corr.append([close_v, df_a['t'][b]])
        a += 1
        b +=1

    open_corr_df = pd.DataFrame(open_corr)
    close_corr_df = pd.DataFrame(close_corr)

    open_corr_df[1] = pd.to_datetime(open_corr_df[1])
    close_corr_df[1] = pd.to_datetime(close_corr_df[1])



    # Set the date column as the index
    open_corr_df.set_index(1, inplace=True)
    close_corr_df.set_index(1, inplace=True)
    # Resample data to daily frequency, excluding weekends
    df_daily_open = open_corr_df.resample('B').mean()
    df_daily_open = df_daily_open[df_daily_open.index.dayofweek < 5]
    df_daily_close = close_corr_df.resample('B').mean()
    df_daily_close = df_daily_close[df_daily_close.index.dayofweek < 5]
    # Define the ARIMA model
    model_open = ARIMA(df_daily_open, order=(1,0,0))
    model_close = ARIMA(df_daily_close, order=(1,0,0))
    # Fit the model
    results_open = model_open.fit()
    results_close=model_close.fit()
    # Make predictions for the next day
    predictions_open = results_open.get_forecast(steps=5)
    predicted_mean_open = predictions_open.predicted_mean
    predicted_conf_open = predictions_open.conf_int()
    predictions_close = results_close.get_forecast(steps=5)
    predicted_mean_close = predictions_close.predicted_mean
    predicted_conf_close = predictions_close.conf_int()

    predicted_conf_open.reset_index(inplace=True)
    predicted_conf_open = predicted_conf_open.rename(columns = {'index':'Dates'})
    predicted_conf_close.reset_index(inplace=True)
    predicted_conf_close = predicted_conf_close.rename(columns = {'index':'Dates'})

    return predicted_conf_open,predicted_conf_close, Stock1['ticker'],Stock2['ticker']