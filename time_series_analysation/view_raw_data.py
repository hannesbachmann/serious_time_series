import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time_series_analysation.data_loader import Loader
from clean_up_data.handle_outliers import handle_to_high_values
from clean_up_data.handle_outliers import handle_shut_down_values


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
        ax1.plot(years[i_y-1].index, years[i_y-1]['P_pool_historical'])
        ax2.plot(years[i_y].index, years[i_y]['P_pool_historical'])
        plt.show()
        pass
    pass


def plot_daily_means(ts):
    # split timeseries into smaller daily dataframe timeseries
    days = [g for n, g in ts.set_index('timestamp').groupby(pd.Grouper(freq='D'))]
    days_means = pd.DataFrame({'timestamp': [g.index[0] for g in days],
                               'P_pool_historical_day_mean': [g.mean()[0] for g in days]}).set_index('timestamp')

    days_means.plot()
    plt.show()
    # low outlier weeks
    # - 2016-04-26T00:00:00 -> bad day need to be corrected
    # - 2018-03-15T00:00:00 -> bad day need to be corrected
    # - 2018-04-12T00:00:00 -> bad day need to be corrected
    # - 2020-09-29T00:00:00 -> bad day need to be corrected
    # - 2020-10-07T00:00:00 -> bad day need to be corrected
    # - 2021-02-10T00:00:00 -> shut down bad measurement data
    # - 2021-02-11T00:00:00 -> shut down bad measurement data
    pass


if __name__ == '__main__':
    L = Loader()
    pool, substations = L.get_data()
    pool = handle_to_high_values(ts=pool[['timestamp', 'P_pool_historical']])
    # plot_daily_means(pool[['timestamp', 'P_pool_historical']])
    # pool = handle_shut_down_values(ts=pool, replacing_method='values')

    plot_time_series_yearly(ts=pool)
    plot_time_series(ts=pool)
    pass
