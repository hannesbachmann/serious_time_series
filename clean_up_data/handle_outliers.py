import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def handle_to_high_values(ts):
    """P_pool_historical measurements between 2022-02-02T00:00:00 and 2022-02-02T23:45:00 show some bad behavior:
    The values are much too high.
    But dividing the power values in this timerange by 200 'fix' this issue quite well."""

    ts['P_pool_historical'] = ts.apply(
        lambda row: row['P_pool_historical'] if row['timestamp'] < pd.to_datetime('2022-02-02T00:00:00') or row[
            'timestamp'] > pd.to_datetime('2022-02-02T23:45:00') else row['P_pool_historical'] / 200, axis=1)

    return ts


def handle_shut_down_values(ts, replacing_method='zeros'):
    """problematic measurement values spots
    between 2021-02-10T00:00:00 and 2021-02-17T00:00:00
    maybe caused by a shut-down.
    values: replace with values from some weeks before
    zeros: replace with zero values
    delete: delete rows"""
    if replacing_method == 'values':
        dt_start = pd.to_datetime('2021-02-10T00:00:00') - pd.Timedelta(days=28)
        dt_end = pd.to_datetime('2021-02-10T00:00:00') - pd.Timedelta(days=7)

        ts1 = ts.set_index('timestamp')
        week_before = ts1[dt_start:dt_end]
        week_before = week_before.reset_index()
        week_before['timestamp'] = week_before['timestamp'].apply(lambda date: date + pd.Timedelta(days=14))
        week_before = week_before.set_index('timestamp')

        ts['P_pool_historical'] = ts.apply(
            lambda row: row['P_pool_historical'] if row['timestamp'] < pd.to_datetime('2021-02-10T00:00:00') - pd.Timedelta(days=7) or row[
                'timestamp'] > pd.to_datetime('2021-02-10T00:00:00') + pd.Timedelta(days=7) else week_before['P_pool_historical'][row['timestamp']], axis=1)
    elif replacing_method == 'zeros':
        ts['P_pool_historical'] = ts.apply(
            lambda row: row['P_pool_historical'] if row['timestamp'] < pd.to_datetime(
                '2021-02-10T00:00:00') or row['timestamp'] > pd.to_datetime(
                '2021-02-10T00:00:00') + pd.Timedelta(days=7) else 0,
            axis=1)
    elif replacing_method == 'delete':
        ts = ts.set_index('timestamp')
        ts_before = ts[:pd.to_datetime('2021-02-10T00:00:00')]
        ts_after = ts[pd.to_datetime('2021-02-10T00:00:00') + pd.Timedelta(days=7):]
        ts = ts_before.append(ts_after).reset_index()
    return ts
