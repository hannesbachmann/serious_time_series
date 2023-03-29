import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time_series_analysation.data_loader import Loader


def prepare_timestamp_features(ts):
    df = ts.copy()
    df['day_of_week'] = df['timestamp'].apply(lambda d: d.day_of_week)
    df['minute_15'] = df['timestamp'].apply(lambda d: (d.hour * 4) + (d.minute / 15))
    return df


def smooth_ts(ts, column, value=2):
    ts[column] = ts[column].rolling(value).mean()
    return ts


def cut_low_values(ts):
    """cut all values of the time series under the rolling mean over one day"""
    ts['roll_mean'] = ts['P_pool_historical'].rolling(24*4*7).mean()
    ts = ts[ts['P_pool_historical'] >= ts['roll_mean']]
    return ts


if __name__ == '__main__':
    L = Loader()
    time_series = L.get_pool_and_temperature_static().copy()

    df = cut_low_values(ts=time_series)
    # df = prepare_timestamp_features(ts=time_series)

    # # STORE STORE STORE
    # try:
    #     df.to_csv('../measured_values/pool_2015_2022_training.csv', sep='|')
    #     print('store dataframe was successful')
    # except:
    #     print('store dataframe failed')
    pass
