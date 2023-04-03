"""analysis of trend and seasonality"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from time_series_analysation.data_loader import Loader
from time_series_analysation.power_temperature_analysis import plot_power_and_temperature


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
    # result: there is not really a trend over all the years
    pass


def calc_seasonality_power(ts):
    """filter seasonality"""
    # separate weekly trend
    decomposed_week = seasonal_decompose(ts['P_pool_historical'], model='additive', period=4*6*5) # period=4*24*7)
    ts['seasonal_week'] = decomposed_week.seasonal
    week_mean = decomposed_week.trend.mean() + decomposed_week.resid.mean()
    ts['without_seasonal'] = ts['P_pool_historical'] - decomposed_week.seasonal     # = trend + resid for week
    # separate yearly trend
    decomposed_year = seasonal_decompose(ts['without_seasonal'], model='additive', period=4*6*5*53)# period=4 * 24 * 365)
    ts['seasonal_year'] = decomposed_year.seasonal
    ts['without_seasonal_year'] = ts['without_seasonal'] - decomposed_year.seasonal     # = trend + resid for year
    year_mean = decomposed_year.trend.mean() + decomposed_year.resid.mean()
    df = ts[['P_pool_historical']]
    # composition evaluation using the seasonality for week and year and the means of trend and resid
    df['composed'] = ts['seasonal_week'] + ts['seasonal_year'] + year_mean
    df['seasonal'] = ts['seasonal_week'] + ts['seasonal_year']
    df['without_seasonal'] = ts['without_seasonal_year']

    # df.plot()
    # plt.show()
    # result: there is a strong seasonality for the periods week and year
    # time_series = trend + season + resid
    # = > time_series - season = trend + resid
    #
    # time_series - season = season_y + trend_y + resid_y
    # = > time_series - season - season_y = trend_y + resid_y
    #
    # = > time_series = trend_y + resid_y + season + season_y
    return df


def calc_seasonality_temperature(ts):
    """filter seasonality"""
    # separate weekly trend
    decomposed_week = seasonal_decompose(ts['T_historical'], model='additive', period=4 * 24 * 7)
    ts['seasonal_week'] = decomposed_week.seasonal
    week_mean = decomposed_week.trend.mean() + decomposed_week.resid.mean()
    ts['without_seasonal'] = ts['T_historical'] - decomposed_week.seasonal  # = trend + resid for week
    # separate yearly trend
    decomposed_year = seasonal_decompose(ts['without_seasonal'], model='additive', period=4 * 24 * 365)
    ts['seasonal_year'] = decomposed_year.seasonal
    ts['without_seasonal_year'] = ts['without_seasonal'] - decomposed_year.seasonal  # = trend + resid for year
    year_mean = decomposed_year.trend.mean() + decomposed_year.resid.mean()
    df = ts[['T_historical']]
    # composition evaluation using the seasonality for week and year and the means of trend and resid
    df['composed'] = ts['seasonal_week'] + ts['seasonal_year'] + year_mean
    df['seasonal'] = ts['seasonal_week'] + ts['seasonal_year']
    df['without_seasonal'] = ts['without_seasonal_year']

    # df.plot()
    # plt.show()
    # result: there is a strong seasonality for the periods week and year
    return df


def compare_static_means(ts):
    """compare the decomposed static temperature and power values"""
    p_mean = df['P_pool_historical'].mean()
    t_mean = df['T_historical'].mean()
    ts['p_over_mean'] = ts['P_pool_historical'].apply(lambda x: 10 if x >= p_mean else -10)
    ts['t_under_mean'] = ts['T_historical'].apply(lambda x: 10 if x <= t_mean else -10)
    ts['corr'] = ts.apply(lambda row: 1 if row['p_over_mean'] == row['t_under_mean'] else 0, axis=1)
    # ts.set_index('timestamp')[['p_over_mean', 't_under_mean']].plot()
    correct = ts['corr'].mean()
    ts.set_index('timestamp')[['corr']].plot()
    plt.show()

    pass


if __name__ == '__main__':
    L = Loader()
    time_series = L.get_pool_and_temperature()
    time_series['P_pool_historical'] = time_series['P_pool_historical'].interpolate(method='linear')

    # calc_linear_trend(ts=time_series[['P_pool_historical']])
    df_power = calc_seasonality_power(ts=time_series.set_index('timestamp')[['P_pool_historical']])
    df_temperature = calc_seasonality_temperature(ts=time_series.set_index('timestamp')[['T_historical']])
    df = df_power[['composed']].copy()
    df['T_historical_train'] = df_temperature['without_seasonal']
    df['P_pool_historical_train'] = df_power['without_seasonal']
    df['P_pool_historical_seasonal'] = df_power['seasonal']
    df['T_historical_seasonal'] = df_temperature['seasonal']
    df['T_historical'] = time_series.set_index('timestamp')['T_historical']
    df['P_pool_historical'] = time_series.set_index('timestamp')['P_pool_historical']
    df = df.drop('composed', axis=1)

    # plot_power_and_temperature(ts=df[['T_historical', 'P_pool_historical']].reset_index())

    # STORE STORE STORE
    # try:
    #     df.to_csv('../measured_values/pool_2015_2022_handled_outliers_static.csv', sep='|')
    #     print('store dataframe was successful')
    # except:
    #     print('store dataframe failed')

    compare_static_means(ts=df[['T_historical', 'P_pool_historical']].reset_index())
    pass
