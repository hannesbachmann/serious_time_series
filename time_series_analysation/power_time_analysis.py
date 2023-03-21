import pandas as pd
import matplotlib.pyplot as plt
from time_series_analysation.data_loader import Loader
from power_temperature_analysis import plot_power_and_temperature


def plot_work_days(ts):
    ts['work_day'] = ts['timestamp'].apply(lambda d: 1 if d.day_of_week < 5 else 0)
    ts = ts[ts['work_day'] == 1]

    ts['day_of_week'] = ts['timestamp'].apply(lambda d: d.hour)
    days = ts[['T_historical', 'P_pool_historical', 'day_of_week']].groupby(['day_of_week']).mean()
    days['timestamp'] = list(range(days.shape[0]))
    days['P_pool_historical'] = days['P_pool_historical'].round(2)
    max_power = days['P_pool_historical'].max()     # ~14000 W
    min_power = days['P_pool_historical'].min()     # ~1700 W
    max_power_timestamp = list(days[days['P_pool_historical'] == max_power]['timestamp'])[0]    # 07:00 morning
    min_power_timestamp = list(days[days['P_pool_historical'] == min_power]['timestamp'])[0]    # 02:00 night

    plot_power_and_temperature(ts=days)
    pass


def plot_weekend(ts):
    ts['work_day'] = ts['timestamp'].apply(lambda d: 1 if d.day_of_week < 5 else 0)
    ts = ts[ts['work_day'] == 0]
    ts['day_of_week'] = ts['timestamp'].apply(lambda d: d.hour)
    days = ts[['T_historical', 'P_pool_historical', 'day_of_week']].groupby(['day_of_week']).mean()
    days['timestamp'] = list(range(days.shape[0]))
    days['P_pool_historical'] = days['P_pool_historical'].round(2)
    max_power = days['P_pool_historical'].max()  # ~8900 W
    min_power = days['P_pool_historical'].min()  # ~2700 W
    max_power_timestamp = list(days[days['P_pool_historical'] == max_power]['timestamp'])[0]  # 18:00 evening
    min_power_timestamp = list(days[days['P_pool_historical'] == min_power]['timestamp'])[0]  # 03:00 night

    plot_power_and_temperature(ts=days)
    pass


if __name__ == '__main__':
    # sep work day and weekend
    L = Loader()
    time_series = L.get_pool_and_temperature()

    # plot_work_days(ts=time_series)
    plot_weekend(ts=time_series)
