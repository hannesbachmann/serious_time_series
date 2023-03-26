"""analysis of trend and saisonality"""
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
    axes[0].set_title('%.2f mehr Passagiere pro Monat (linearer Trend)' % regr.coef_[0])
    axes[0].plot(x, y)
    axes[0].plot(x, trend_pred, color='black')
    axes[0].set_xlabel('Zeit $t$')
    axes[0].set_ylabel('Passagiere')

    axes[1].set_title('Trendbereinigte Zahl der Passagiere')
    axes[1].plot(x, air_passengers_detrended)
    axes[1].set_xlabel('Zeit $t$')
    axes[1].set_ylabel('Passagiere')
    plt.show()
    # result: there is not really a trend over the years
    pass


def calc_seasonality(ts):
    """filter seasonality"""
    decomposed = seasonal_decompose(ts['P_pool_historical'], model='additive', period=4*24*7)
    decomposed.plot()
    plt.show()

    ts['seasonal'] = decomposed.seasonal
    ts['without_seasonal'] = ts['P_pool_historical'] - (decomposed.seasonal + 10000)

    decomposed_year = seasonal_decompose(ts['without_seasonal'], model='additive', period=4*24*365)
    ts['without_seasonal_year'] = ts['without_seasonal'] - decomposed_year.seasonal

    # f, axes = plt.subplots(1, 2, figsize=(12, 4))
    # axes[0].set_title('Monatliche Abweichung vom Mittelwert')
    # axes[0].bar(['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'], monthly_means)
    # axes[0].set_xlabel('Zeit $t$')
    # axes[0].set_ylabel('Passagiere')
    #
    # axes[1].set_title('Trend- und saisonbereinigte Zahl der Passagiere')
    # axes[1].plot(x, air_passengers_detrended_unseasonal)
    # axes[1].set_xlabel('Zeit $t$')
    # axes[1].set_ylabel('Passagiere')
    # plt.show()

    pass


if __name__ == '__main__':
    L = Loader()
    time_series = L.get_pool_and_temperature()
    time_series['P_pool_historical'] = time_series['P_pool_historical'].interpolate(method='linear')

    # calc_linear_trend(ts=time_series[['P_pool_historical']])
    calc_seasonality(ts=time_series.set_index('timestamp')[['P_pool_historical']])
    pass
