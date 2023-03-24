"""analysis correlations between previous values with the current value"""
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import linregress
from time_series_analysation.data_loader import Loader


def plot_day_before_and_today(ts):
    ts['P_pool_historical'] = ts['P_pool_historical'].round(2)
    p_prev = [0]
    for i in range(1, ts.shape[0]):
        p_prev.append(ts['P_pool_historical'][i-1])
    ts['P_prev'] = p_prev

    pass


def autocorrelation_power(ts, freq):
    dfs_mean = split_ts(ts, freq)

    dfs_mean[['P_pool_historical']].plot()
    plot_acf(dfs_mean['P_pool_historical'], lags=dfs_mean.shape[0]-1)
    plot_pacf(dfs_mean['P_pool_historical'], lags=int(dfs_mean.shape[0] / 2)-1)
    plt.show()
    # positive: month, week, day
    # negative: year, (quarter), hour, minute
    pass


def autocorrelation_temperature(ts, freq):
    dfs_mean = split_ts(ts, freq)

    dfs_mean[['T_historical']].plot()
    plot_acf(dfs_mean['T_historical'], lags=dfs_mean.shape[0]-1)
    plot_pacf(dfs_mean['T_historical'], lags=int(dfs_mean.shape[0] / 2)-1)
    plt.show()
    # positive: month, week, day
    # negative: year, (quarter), hour, minute
    pass


def split_ts(ts, freq):
    # split timeseries into smaller dataframe timeseries based on the given frequency
    dfs = [g for n, g in ts.set_index('timestamp').groupby(pd.Grouper(freq=freq))]  # or 'Q' for quarter
    # create dataframe to store the means for every step based on the given frequency
    dfs_mean = pd.DataFrame({'timestamp': [g.index[0] for g in dfs],
                             f'P_pool_historical': [g['P_pool_historical'].mean() for g in dfs],
                             f'T_historical': [g['T_historical'].mean() for g in dfs]})
    return dfs_mean


if __name__ == '__main__':
    L = Loader()
    time_series = L.get_pool_and_temperature()
    autocorrelation_power(ts=time_series, freq='D')
    autocorrelation_temperature(ts=time_series, freq='D')
    # plot_day_before_and_today(ts=time_series)

    pass