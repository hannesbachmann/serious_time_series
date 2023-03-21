import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from time_series_analysation.data_loader import Loader
from clean_up_data.handle_outliers import handle_to_high_values
from clean_up_data.handle_outliers import handle_shut_down_values
from clean_up_data.handle_outliers import handle_low_day_values


def plot_time_series(ts):
    ts = ts.set_index('timestamp')
    ts.plot()
    # Giving title to the graph
    plt.title(f'{list(ts.columns)} power/time')

    # rotating the x-axis tick labels at 30degree
    # towards right
    plt.xticks(rotation=10, ha='right')

    # Giving x and y label to the graph
    plt.xlabel('time')
    plt.ylabel('power in W')
    plt.show()
    pass


def plot_time_series_yearly(ts):
    # split timeseries into smaller yearly dataframe timeseries
    years = [g for n, g in ts.set_index('timestamp').groupby(pd.Grouper(freq='Y'))]

    for i_y in range(1, len(years)):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        ax1.plot(years[i_y - 1].index, years[i_y - 1]['P_pool_historical'])
        ax2.plot(years[i_y].index, years[i_y]['P_pool_historical'])
        plt.show()
        pass
    # low outliers
    # - 2020-03-15T00:00:00 - 2020-05-10T00:00:00 a bunch of too low weeks (first corona lockdown and some around)
    pass


def plot_daily_means(ts):
    # split timeseries into smaller daily dataframe timeseries
    days = [g for n, g in ts.set_index('timestamp').groupby(pd.Grouper(freq='D'))]
    # create dataframe to store the means for every day
    days_means = pd.DataFrame({'timestamp': [g.index[0] for g in days],
                               'P_pool_historical_day_mean': [g.mean()[0] for g in days]}).set_index('timestamp')

    days_means.plot()
    plt.show()
    # outlier days with too low values
    # - 2016-04-26T00:00:00 -> too low values for that day, need to be corrected (no holiday)
    # - 2018-03-15T00:00:00 -> too low values for that day, need to be corrected (no holiday)
    # - 2018-04-12T00:00:00 -> too low values for that day, need to be corrected (no holiday)
    # - 2020-09-29T00:00:00 -> too low values for that day, need to be corrected (no holiday)
    # - 2020-10-07T00:00:00 -> too low values for that day, need to be corrected (no holiday)
    # - 2020-05-01T00:00:00 -> low values for that day (Workers day)
    # - 2020-05-03T00:00:00 -> low values for that day (sunday)
    # - 2020-05-10T00:00:00 -> low values for that day (sunday)
    # - 2021-02-10T00:00:00 -> shut down bad measurement data
    # - 2021-02-11T00:00:00 -> shut down bad measurement data
    pass


def plot_weekly_mean(ts):
    # split timeseries into smaller weekly dataframe timeseries
    weeks = [g for n, g in ts.set_index('timestamp').groupby(pd.Grouper(freq='W'))]
    # create dataframe to store the means for every week
    weeks_mean = pd.DataFrame({'timestamp': [g.index[0] for g in weeks],
                               'P_pool_historical_week_mean': [g.mean()[0] for g in weeks]}).set_index('timestamp')

    weeks_mean.plot()
    plt.show()
    # low outliers
    # - 2020-03-15T00:00:00 - 2020-05-10T00:00:00 a bunch of too low weeks
    #       (first corona lockdown 2020-05-22T00:00:00 - 2020-04-04T00:00:00 and some weeks before/after)
    pass


def plot_yearly_means(ts):
    # split timeseries into smaller yearly dataframe timeseries
    years = [g for n, g in ts.set_index('timestamp').groupby(pd.Grouper(freq='Y'))]  # or 'Q' for quarter
    # create dataframe to store the means for every year
    years_mean = pd.DataFrame({'timestamp': [g.index[0] for g in years], 'P_pool_historical_year_mean':
        [g.mean()[0] for g in years]}).set_index('timestamp')

    years_mean.plot()
    plt.show()
    # high values for 2018
    # low values for 2020
    # 2022 is not complete, so this year will eventually be cut
    # quarters look relatively normal
    pass


if __name__ == '__main__':
    L = Loader()
    pool, substations, temperature = L.get_data()
    time_series = L.get_pool_and_temperature().copy()

    # plot_yearly_means(ts=pool[['timestamp', 'P_pool_historical']])
    # plot_weekly_mean(ts=pool[['timestamp', 'P_pool_historical']])
    # plot_daily_means(ts=pool[['timestamp', 'P_pool_historical']])
    # plot_time_series_yearly(ts=pool[['timestamp', 'P_pool_historical']])
    plot_time_series(ts=pool)

    # STORE STORE STORE
    # try:
    #     pool.to_csv('hier könnte ihr path stehen', sep='|')
    #     print('store dataframe was successful')
    # except:
    #     print('store dataframe failed')
    pass
