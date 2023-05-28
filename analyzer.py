import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor
import concurrent
import multiprocessing


class StockDataAnalyzer:

    polygon_api_key = "" 

    def __init__(self, stock_names_file):
        self.stock_names = pd.read_csv(stock_names_file)['0']
        self.data = []
        self.data_complete = []

    def get_data(self, ticker):
        try:
            time.sleep(0.01)  # sleep for 0.01 seconds
            current_date = str(datetime.today())[:10]
            one_year_ago = str(datetime.today() - timedelta(days=365))[:10]
            URL = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{one_year_ago}/{current_date}?apiKey={self.polygon_api_key}'
            api_result = requests.get(url=URL)
            result_json = api_result.json()

            if 'results' in result_json:
                for i in range(len(result_json['results'])):
                    unix_time = int(result_json['results'][i]['t']) / 1000
                    datetime_obj = datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
                    result_json['results'][i]['t'] = datetime_obj

                return result_json
            else:
                return None
        except Exception as e:
            print(f"Error occurred for ticker {ticker}: {e}")
            return None

    def fetch_data(self):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_data, ticker) for ticker in self.stock_names]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    self.data.append(result)

    def filter_complete_data(self):
        for i in range(len(self.data)):
            if self.data[i]['queryCount'] == 252:
                self.data_complete.append(self.data[i])

    def get_all_correlations(self, stock1, stock2):
        stock1_ = stock1['ticker']
        stock2_ = stock2['ticker']
        if stock1_ != stock2_:
            try:
                df1 = pd.DataFrame.from_dict(stock1['results'], orient='columns', dtype=None, columns=None)
                df2 = pd.DataFrame.from_dict(stock2['results'], orient='columns', dtype=None, columns=None)
                c, p = stats.pearsonr(df1.dropna()['c'], df2.dropna()['c'])
                open_v = c
                c, p = stats.pearsonr(df1.dropna()['o'], df2.dropna()['o'])
                close_v = c

                return [stock1_, stock2_, open_v, close_v]
            except ValueError:
                pass

    def analyze_correlations(self):
        get_correlation = []
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = [executor.submit(self.get_all_correlations, self.data_complete[i], self.data_complete[j])
                       for i in range(len(self.data_complete))
                       for j in range(len(self.data_complete)) if i != j
                       ]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                get_correlation.append(res)

        new_res = [i for i in get_correlation if i is not None]
        
        self.save_correlation_results(new_res)

    def save_correlation_results(self, new_res):
        df = pd.DataFrame(new_res)
        try:
            df.to_csv(f'Corr_{datetime.today()}.csv')
        except:
            df.to_csv('Corr.csv')

analyzer = StockDataAnalyzer('strongbond/all_us_stocks.csv')
analyzer.fetch_data()
analyzer.filter_complete_data()
analyzer.analyze_correlations()





