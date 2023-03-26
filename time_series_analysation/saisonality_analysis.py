"""analysis of trend and saisonality"""
import pandas as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
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


if __name__ == '__main__':
    L = Loader()
    time_series = L.get_pool_and_temperature()
    time_series['P_pool_historical'] = time_series['P_pool_historical'].interpolate(method='linear')

    calc_linear_trend(ts=time_series[['P_pool_historical']])

    pass
