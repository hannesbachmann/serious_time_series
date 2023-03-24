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

    c_power_time = days['P_pool_historical'].corr(days['timestamp'], method='pearson')
    c_power_temperature = days['P_pool_historical'].corr(days['T_historical'], method='pearson')
    df = days.copy()

    df['min'] = df['P_pool_historical'][(df['P_pool_historical'].shift(1) > df['P_pool_historical']) & (df['P_pool_historical'].shift(-1) > df['P_pool_historical'])]
    df['max'] = df['P_pool_historical'][(df['P_pool_historical'].shift(1) < df['P_pool_historical']) & (df['P_pool_historical'].shift(-1) < df['P_pool_historical'])]
    df = df.fillna(0)

    df[['P_pool_historical', 'min', 'max']].plot()
    plt.show()

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

    c_power_time = days['P_pool_historical'].corr(days['timestamp'], method='pearson')
    c_power_temperature = days['P_pool_historical'].corr(days['T_historical'], method='pearson')
    plot_power_and_temperature(ts=days)
    pass


def separate_day_into_segments(ts):
    ts['hour'] = ts['timestamp'].apply(lambda d: d.hour)
    ts['year'] = ts['timestamp'].apply(lambda d: d.year)
    ts['day_of_year'] = ts['timestamp'].apply(lambda d: d.dayofyear)
    ts['day_section'] = ts['timestamp'].apply(lambda d: 'night' if (d.hour >= 22) or (d.hour < 5) else 'rush_hour' if (d.hour >= 5) & (d.hour < 11) else 'afternoono')
    days_with_section = ts.groupby(['day_of_year', 'day_section', 'year']).max().set_index('timestamp').sort_index().reset_index()
    days_with_section['year'] = days_with_section['timestamp'].apply(lambda d: d.year)
    days_with_section['day_of_year'] = days_with_section['timestamp'].apply(lambda d: d.dayofyear)
    day_with_section = days_with_section.groupby(['year', 'day_of_year']).aggregate(list)

    day_with_section['work_day'] = day_with_section.apply(lambda row: 1 if row['timestamp'][0].day_of_week < 5 else 0,
                                                          axis=1)
    day_with_section['correct'] = day_with_section.apply(lambda row: 0 if row['P_pool_historical'][2] > row['P_pool_historical'][0] and row['work_day'] == 1 else 1,
                                                         axis=1)

    incorrect = day_with_section[day_with_section['correct'] == 0]
    incorrect = incorrect.apply(pd.Series.explode)

    incorrect_times = list(incorrect['timestamp'])

    ts['incorrect'] = ts.apply(lambda row: 30000 if row['timestamp'] in incorrect_times else 0, axis=1)
    # ts.set_index('timestamp')[['P_pool_historical', 'T_historical', 'incorrect']].plot()
    # plt.show()

    # night from 2015-06-07 to 2015-06-07 --> high max
    # night from 2015-06-21 to 2015-06-22 --> high max
    # and some others: seems to be an effect that only appears sometimes at the weekend/holidays

    pass


if __name__ == '__main__':
    pool_data = pd.read_csv('../measured_values/pool_2015_2022.csv', delimiter='|')
    pool_data['timestamp'] = pd.to_datetime(pool_data['timestamp'])
    # split timeseries into smaller dataframe timeseries based on the given frequency
    dfs = [g for n, g in pool_data.set_index('timestamp').groupby(pd.Grouper(freq='Y'))]  # or 'Q' for quarter

    # sep work day and weekend
    L = Loader()
    time_series = L.get_pool_and_temperature()

    separate_day_into_segments(ts=time_series)

    # plot_work_days(ts=time_series)
    plot_weekend(ts=time_series)
