"""analysis of trend and seasonality"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from time_series_analysation.data_loader import Loader


def calc_linear_trend(ts):
    """filter a linear trend using linear regression on power values"""
    x = np.arange(1, ts.shape[0] + 1).reshape(-1, 1)
    y = ts.values

    regr = LinearRegression()
    regr.fit(x, y)
    trend_pred = regr.predict(x)
    air_passengers_detrended = ts - trend_pred

    f, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].set_title('%.2f more power (linear trend)' % regr.coef_[0])
    axes[0].plot(x, y)
    axes[0].plot(x, trend_pred, color='black')
    axes[0].set_xlabel('time $t$')
    axes[0].set_ylabel('power')

    axes[1].set_title('power without trend')
    axes[1].plot(x, air_passengers_detrended)
    axes[1].set_xlabel('time $t$')
    axes[1].set_ylabel('power')
    plt.show()
    # result: there is not really a trend over the years
    pass


def calc_seasonality(ts):
    """filter seasonality"""
    # separate weekly trend
    decomposed_week = seasonal_decompose(ts['P_pool_historical'], model='additive', period=4*24*7)
    ts['seasonal_week'] = decomposed_week.seasonal
    week_mean = decomposed_week.trend.mean() + decomposed_week.resid.mean()
    ts['without_seasonal'] = ts['P_pool_historical'] - decomposed_week.seasonal     # = trend + resid for week
    # separate yearly trend
    decomposed_year = seasonal_decompose(ts['without_seasonal'], model='additive', period=4 * 24 * 365)
    ts['seasonal_year'] = decomposed_year.seasonal
    ts['without_seasonal_year'] = ts['without_seasonal'] - decomposed_year.seasonal     # = trend + resid for year
    year_mean = decomposed_year.trend.mean() + decomposed_year.resid.mean()
    df = ts[['P_pool_historical', 'without_seasonal_year']]
    # composition evaluation using the seasonal effects for a week and a year and the means of trend and resid
    df['composed'] = ts['seasonal_week'] + ts['seasonal_year'] + year_mean

    decomposed_week.plot()
    plt.show()

    pass


if __name__ == '__main__':
    L = Loader()
    time_series = L.get_pool_and_temperature()
    time_series['P_pool_historical'] = time_series['P_pool_historical'].interpolate(method='linear')

    # calc_linear_trend(ts=time_series[['P_pool_historical']])
    calc_seasonality(ts=time_series.set_index('timestamp')[['P_pool_historical']])
    pass
