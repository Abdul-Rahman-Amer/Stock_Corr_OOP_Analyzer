import requests
from datetime import datetime, timedelta
import time
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns

def get_data_3_years(ticker):
    try:
        time.sleep(0.01) # sleep for 0.01 seconds
        current_date = str(datetime.today())[:10]
        one_year_ago = str(datetime.today() - timedelta(days=365*3))[:10]
        URL = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{one_year_ago}/{current_date}?apiKey=ZHbp7SluVQgGgSGl64K_kEnW0HbYoqqj'
        api_result = requests.get(url=URL)
        result_json = api_result.json()

        if 'results' in result_json:
            for i in range(0,len(result_json['results'])):
                unix_time=int(result_json['results'][i]['t'])/1000
                datetime_obj = datetime.fromtimestamp(unix_time ).strftime('%Y-%m-%d %H:%M:%S')
                result_json['results'][i]['t']=datetime_obj

            return result_json
        else:
            return None
    except Exception as e:
        print(f"Error occurred for ticker {ticker}: {e}")
        return None
a=get_data_3_years('MSFT')
b = get_data_3_years('TSLA')

df_a = pd.DataFrame(a['results'])
df_b = pd.DataFrame(b['results'])

open_corr = []
close_corr= []

a = 0
b = 252
df_len=len(df_a)
while b < df_len:

    c,p = stats.pearsonr(df_a[a:b].dropna()['c'], df_b[a:b].dropna()['c'])
    open_v = c
    c,p = stats.pearsonr(df_a[a:b].dropna()['o'], df_b[a:b].dropna()['o'])
    close_v = c
    open_corr.append([open_v,df_a['t'][b]])
    close_corr.append([close_v, df_a['t'][b]])
    a += 1
    b +=1

open_corr_df = pd.DataFrame(open_corr)
close_corr_df = pd.DataFrame(close_corr)


# Set the date column as the index
open_corr_df[1] = pd.to_datetime(open_corr_df[1])
open_corr_df.set_index(1, inplace=True)


# Resample data to daily frequency, excluding weekends
df_daily = open_corr_df.resample('D').mean()
df_daily = df_daily[df_daily.index.dayofweek < 5]

# Define the ARIMA model
model = ARIMA(df_daily, order=(1,0,0))

# Fit the model
results = model.fit()

# Make predictions for the next day
predictions = results.get_forecast(steps=5)
predicted_mean = predictions.predicted_mean
predicted_conf = predictions.conf_int()

residuals = results.resid
residuals_rolling = residuals.rolling(window=7).mean()

# Plotting residuals with rolling average
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.plot(residuals_rolling, color='r', label='Residuals (Rolling Avg)')
plt.axhline(y=0, color='black', linestyle='-')
plt.legend()
plt.title("MSFT vs TSLA Residuals")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.show()

import statsmodels.api as sm
# Plotting QQ plot of residuals
plt.figure(figsize=(6, 6))
sm.qqplot(residuals, line='s', color='b')
plt.title("QQ Plot of Residuals")
plt.show()

# Print AIC and BIC
print("AIC:", results.aic)
print("BIC:", results.bic)

print(predicted_conf)