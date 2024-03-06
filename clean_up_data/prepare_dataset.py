import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time_series_analysation.data_loader import Loader
from time_series_analysation.power_time_analysis import separate_day_into_segments
from time_series_analysation.seasonality_analysis import calc_seasonality_power
from prediction.random_forest_reg import prepare_prediction


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


def only_rush_hour(ts):
    ts = separate_day_into_segments(ts)
    ts = ts[ts['day_section'] == 'rush_hour']
    ts['day_of_week'] = ts['timestamp'].apply(lambda d: d.day_of_week)
    ts = ts[ts['day_of_week'] < 5]
    # plot power values for rush hours
    # ts.set_index('timestamp')[['P_pool_historical']].plot()
    # plt.show()

    # using only the rush hour for prediction did not improve the prediction at all
    return ts


if __name__ == '__main__':
    L = Loader()
    time_series = L.get_pool_and_temperature_static().copy()

    df = only_rush_hour(ts=time_series)
    df_seasonal = calc_seasonality_power(ts=df)
    df_seasonal['timestamp'] = list(df['timestamp'])
    df_train = prepare_timestamp_features(ts=df_seasonal)   #.set_index('timestamp')
    df_train['T_historical'] = df['T_historical']
    df_train['T_historical_seasonal'] = df['T_historical_seasonal']
    df_train['day_of_year'] = df['day_of_year']
    features = ['T_historical', 'minute_15', 'day_of_week', 'day_of_year']
    # prepare_prediction(time_series=df_train.set_index('timestamp'), features=features)

    pass

    # df = cut_low_values(ts=time_series)
    # df = prepare_timestamp_features(ts=time_series)

    # # STORE STORE STORE
    # try:
    #     df.to_csv('../measured_values/pool_2015_2022_training.csv', sep='|')
    #     print('store dataframe was successful')
    # except:
    #     print('store dataframe failed')
    pass
