import pandas as pd
import matplotlib.pyplot as plt


def handle_to_high_values(ts):
    """P_pool_historical measurements between 2022-02-02T00:00:00 and 2022-02-02T23:45:00 show some bad behavior:
    The values are much too high.
    But dividing the power values in this timerange by 200 'fix' this issue quite well."""

    if ts['timestamp'][0] <= pd.to_datetime('2022-02-02T00:00:00') <= pd.to_datetime(
            '2022-02-02T00:00:00') + pd.Timedelta(days=1) <= ts['timestamp'][ts.shape[0] - 1]:
        ts['P_pool_historical'] = ts.apply(
            lambda row: row['P_pool_historical'] if row['timestamp'] < pd.to_datetime('2022-02-02T00:00:00') or row[
                'timestamp'] > pd.to_datetime('2022-02-02T23:45:00') else row['P_pool_historical'] / 200, axis=1)

    return ts


def handle_shut_down_values(ts, replacing_method='zeros'):
    """problematic measurement values spots
    between 2021-02-10T00:00:00 and 2021-02-17T00:00:00
    maybe caused by a shut-down.
    values: replace with values from some weeks before
    zeros:  replace with zero values
    delete: delete rows

    REMEMBER: when map other features than dates to the dataset, values mode should not be used"""
    if ts['timestamp'][0] <= pd.to_datetime('2021-02-10T00:00:00') - pd.Timedelta(days=28) <= pd.to_datetime(
            '2021-02-10T00:00:00') + pd.Timedelta(days=14) <= ts['timestamp'][ts.shape[0] - 1]:
        if replacing_method == 'values':
            dt_start = pd.to_datetime('2021-02-10T00:00:00') - pd.Timedelta(days=28)
            dt_end = pd.to_datetime('2021-02-10T00:00:00') - pd.Timedelta(days=7)

            ts1 = ts.set_index('timestamp')
            week_before = ts1[dt_start:dt_end]
            week_before = week_before.reset_index()
            week_before['timestamp'] = week_before['timestamp'].apply(lambda date: date + pd.Timedelta(days=14))
            week_before = week_before.set_index('timestamp')

            ts['P_pool_historical'] = ts.apply(
                lambda row: row['P_pool_historical'] if row['timestamp'] < pd.to_datetime(
                    '2021-02-10T00:00:00') - pd.Timedelta(days=7) or row[
                                                            'timestamp'] > pd.to_datetime(
                    '2021-02-10T00:00:00') + pd.Timedelta(days=7) else week_before['P_pool_historical'][
                    row['timestamp']],
                axis=1)
            ts['T_historical'] = ts.apply(
                lambda row: row['T_historical'] if row['timestamp'] < pd.to_datetime(
                    '2021-02-10T00:00:00') - pd.Timedelta(days=7) or row[
                                                       'timestamp'] > pd.to_datetime(
                    '2021-02-10T00:00:00') + pd.Timedelta(days=7) else week_before['T_historical'][row['timestamp']],
                axis=1)
        elif replacing_method == 'zeros':
            ts['P_pool_historical'] = ts.apply(
                lambda row: row['P_pool_historical'] if row['timestamp'] < pd.to_datetime(
                    '2021-02-10T00:00:00') or row['timestamp'] > pd.to_datetime(
                    '2021-02-10T00:00:00') + pd.Timedelta(days=7) else 0,
                axis=1)
            ts['T_historical'] = ts.apply(
                lambda row: row['T_historical'] if row['timestamp'] < pd.to_datetime(
                    '2021-02-10T00:00:00') or row['timestamp'] > pd.to_datetime(
                    '2021-02-10T00:00:00') + pd.Timedelta(days=7) else 0,
                axis=1)
        elif replacing_method == 'delete':
            ts = ts.set_index('timestamp')
            ts_before = ts[:pd.to_datetime('2021-02-10T00:00:00')]
            ts_after = ts[pd.to_datetime('2021-02-10T00:00:00') + pd.Timedelta(days=7):]
            ts = ts_before.append(ts_after).reset_index()
    return ts


def handle_low_day_values(ts, replacing_method='zeros'):
    """outlier days with too low values
    2016-04-26T00:00:00 -> to low values for that day, need to be corrected (no holiday)
    2018-03-15T00:00:00 -> to low values for that day, need to be corrected (no holiday)
    2018-04-12T00:00:00 -> to low values for that day, need to be corrected (no holiday)
    2020-09-29T00:00:00 -> to low values for that day, need to be corrected (no holiday)
    2020-10-07T00:00:00 -> to low values for that day, need to be corrected (no holiday)
    handle in 3 modes:
    values: replace with values from some weeks before
    zeros:  replace with zero values
    delete: delete rows

    REMEMBER: when map other features than dates to the dataset, values mode should not be used"""
    bad_days = ['2016-04-26T00:00:00',
                '2018-03-15T00:00:00',
                '2018-04-12T00:00:00',
                '2020-09-29T00:00:00',
                '2020-10-07T00:00:00']
    for day in bad_days:
        if ts['timestamp'][0] <= pd.to_datetime(day) <= ts['timestamp'][ts.shape[0] - 1]:
            print(f'correcting day: {day}')
            if replacing_method == 'values':
                dt_start = pd.to_datetime(day) - pd.Timedelta(days=1)
                dt_end = pd.to_datetime(day)

                ts1 = ts.set_index('timestamp')
                day_before = ts1[dt_start:dt_end]
                day_before = day_before.reset_index()
                day_before['timestamp'] = day_before['timestamp'].apply(lambda date: date + pd.Timedelta(days=1))
                day_before = day_before.set_index('timestamp')

                ts['P_pool_historical'] = ts.apply(
                    lambda row: row['P_pool_historical'] if row['timestamp'] < pd.to_datetime(day) or row[
                        'timestamp'] > pd.to_datetime(day) + pd.Timedelta(days=1) else day_before['P_pool_historical'][
                        row['timestamp']], axis=1)
                ts['T_historical'] = ts.apply(
                    lambda row: row['T_historical'] if row['timestamp'] < pd.to_datetime(day) or row[
                        'timestamp'] > pd.to_datetime(day) + pd.Timedelta(days=1) else day_before['T_historical'][
                        row['timestamp']], axis=1)
            elif replacing_method == 'zeros':
                ts['P_pool_historical'] = ts.apply(
                    lambda row: row['P_pool_historical'] if row['timestamp'] < pd.to_datetime(day) or row[
                        'timestamp'] > pd.to_datetime(day) + pd.Timedelta(days=1) else 0, axis=1)
                ts['T_historical'] = ts.apply(
                    lambda row: row['T_historical'] if row['timestamp'] < pd.to_datetime(day) or row[
                        'timestamp'] > pd.to_datetime(day) + pd.Timedelta(days=1) else 0, axis=1)
            elif replacing_method == 'delete':
                ts = ts.set_index('timestamp')
                ts_before = ts[:pd.to_datetime(day)]
                ts_after = ts[pd.to_datetime(day) + pd.Timedelta(days=1):]
                ts = ts_before.append(ts_after).reset_index()
    return ts
