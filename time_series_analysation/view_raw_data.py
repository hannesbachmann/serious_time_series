import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time_series_analysation.data_loader import Loader


def plot_time_series(ts=pd.DataFrame({'timestamp': [], 'values': []})):
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


if __name__ == '__main__':
    L = Loader()
    pool, substations = L.get_data()
    plot_time_series(ts=pool[['timestamp', 'P_pool_historical']][:1000])
    pass
