"""power prediction model using a random forest regressor"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from time_series_analysation.data_loader import Loader


def create_and_fit_model(x_train, y_train):
    """x_train: temperature, minute, day_of_week, holiday
    y_train: power
    temperature and power are static"""
    regressor = RandomForestRegressor(n_estimators=20,
                                      max_depth=25,
                                      random_state=42)
    regressor.fit(x_train, y_train)
    return regressor


def predict(regressor, x_pred):
    y_pred = regressor.predict(x_pred)
    return y_pred


def model_evaluation(y_actual, y_pred):
    pass


if __name__ == '__main__':
    L = Loader()
    time_series = L.get_pool_and_temperature_static().copy().set_index('timestamp')
    time_series_valid = time_series[pd.to_datetime('2022-03-20T00:15:00'):]
    time_series_train = time_series[:pd.to_datetime('2022-03-20T00:00:00')]
    # create and train a random forest regressor model
    ran_for_reg = create_and_fit_model(x_train=time_series_train[['T_historical_train']],
                                       y_train=time_series_train['P_pool_historical_train'])
    result = predict(regressor=ran_for_reg,
                     x_pred=time_series_valid[['T_historical_train']])

    total = time_series[['P_pool_historical_train']]
    res = list(time_series_train['P_pool_historical_train'])
    r = list(result)
    for e in r:
        res.append(e)
    total['predicted'] = res

    df = time_series[['P_pool_historical']]
    df['predicted'] = total['predicted'] + time_series['P_pool_historical_seasonal']
    pass
