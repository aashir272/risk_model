import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import os
from pandas.tseries.offsets import BDay
import datetime
# from risk_model.configuration import *

LOOKBACK = 126
RETURN_THRESHOLD = 10
NUM_FACTORS = 50
SHRINK_FACTOR = 0.5
CWD = os.getcwd()
OUTPUT_DIR = f'{CWD}\\data'

class BuildRiskModel(object):

    def __init__(self, date, path=None):
        self.date = date
        self.path = path

    def load_data(self):
        if not self.path:
            path = f'{CWD}\\stock_returns_us.parquet'
        df = pq.read_table(path).to_pandas()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df
    
    def filter_tickers(self, df):
    
        # remove null and outlier returns
        df_not_na = df[(~df.Return.isnull()) & (df.Return.abs() <= RETURN_THRESHOLD)].copy()

        # get tickers that exist on all dates
        df_not_na['Exists'] = 1
        num_dates = len(df_not_na['Date'].unique())
        ticker_date = df_not_na.groupby('Ticker')['Exists'].sum().reset_index()
        filtered_tickers = ticker_date[ticker_date.Exists == num_dates]['Ticker'].tolist()

        # filter dataset
        df_filtered = df_not_na[(df_not_na.Ticker.isin(filtered_tickers))]

        return df_filtered
    
    def build_model(self):

        # load and clean data
        start_date = self.date
        end_date = (start_date - BDay(LOOKBACK)).date()
        df = self.load_data()
        df = self.filter_tickers(df)
        returns = df[(df.Date <= start_date) & (df.Date >= end_date)].copy()

        # build risk model
        panel = (returns
                .pivot(index='Date', columns='Ticker', values='Return')
                .fillna(0.0)
                .sort_index())
        R = panel.T.values
        R = R - R.mean(axis=1, keepdims=True)
        N, T = R.shape

        # factor exposures
        cov = (R@R.T)/(T-1)
        e, V = np.linalg.eigh(cov)
        idx  = e.argsort()[::-1][:NUM_FACTORS]
        e, V  = e[idx], V[:, idx]
        B = V * np.sqrt(e)

        # factor returns
        F = (V.T @ R) / np.sqrt(e)[:, None] 

        # specific variance
        E = R - B @ F
        spec = E.var(axis=1, ddof=1)

        # shrink to overall mean
        target = spec.mean()
        D = SHRINK_FACTOR * spec + (1-SHRINK_FACTOR) * target

        # convert to DataFrame
        tickers = panel.columns.values
        exposures = pd.DataFrame(B, index=tickers).reset_index()
        exposures = exposures.rename(columns={'index': 'Ticker'})
        exposures['Date'] = self.date
        spec = pd.DataFrame(D, index=tickers).reset_index()
        spec = spec.rename(columns={'index': 'Ticker', 0: 'SpecificVariance'})
        spec['Date'] = date

        # save
        exposures.to_parquet(f'{OUTPUT_DIR}\\exposures\\{self.date}.parquet')
        spec.to_parquet(f'{OUTPUT_DIR}\\specific\\{self.date}.parquet')


if __name__ == '__main__':
    start_date = datetime.date(2016, 5, 30)
    end_date = datetime.date(2025, 5, 30)
    current = 2016
    for dt in pd.bdate_range(start_date, end_date):
        date = dt.date()
        print(date)
        builder = BuildRiskModel(date)
        builder.build_model()
