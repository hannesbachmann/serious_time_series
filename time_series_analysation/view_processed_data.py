"""plot processed data to find correlations between power values and other features"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time_series_analysation.data_loader import Loader


def plot_power_and_temperature(ts):
    # plots power values and temperatures over time
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Datetime')
    ax1.set_ylabel('P_pool_historical [P in W]', color='red')
    ax1.plot(ts['timestamp'], ts['P_pool_historical'], color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    # Adding Twin Axes

    ax2 = ax1.twinx()

    ax2.set_ylabel('T_historical [T in °C]', color='blue')
    ax2.plot(ts['timestamp'], ts['T_historical'], color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Show plot

    plt.show()
    pass


def plot_power_against_temperature(ts):
    # plots means of the power values over temperatures
    ts = ts[['T_historical', 'P_pool_historical']].groupby(['T_historical']).mean()
    ts.plot()
    plt.show()
    pass


if __name__ == '__main__':
    L = Loader()
    time_series = L.get_pool_and_temperature().copy()
    # plot_power_and_temperature(ts=time_series)
    plot_power_against_temperature(ts=time_series)
    pass
