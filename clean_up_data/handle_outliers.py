import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def handle_bad_values(ts):
    """P_pool_historical measurements between 2022-02-02T00:00:00 and 2022-02-02T23:45:00 show some bad behavior:
    The values are much too high.
    But dividing the power values in this timerange by 200 'fix' this issue quite well."""

    ts['P_pool_historical'] = ts.apply(
        lambda row: row['P_pool_historical'] if row['timestamp'] < pd.to_datetime('2022-02-02T00:00:00') or row[
            'timestamp'] > pd.to_datetime('2022-02-02T23:45:00') else row['P_pool_historical'] / 200, axis=1)

    return ts
