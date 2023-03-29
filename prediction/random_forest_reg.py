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


def model_evaluation(prediction_and_actual):
    # calculate mean error for each day
    prediction_and_actual['err'] = abs(prediction_and_actual['predicted'] - prediction_and_actual['P_pool_historical'])
    dfs = [g for n, g in prediction_and_actual[['err']].groupby(pd.Grouper(freq='D'))]
    # create dataframe to store the means for every step based on the given frequency
    dfs_mean = pd.DataFrame({'timestamp': [g.index[0] for g in dfs],
                             f'err': [g['err'].mean() for g in dfs]})
    dfs_mean = dfs_mean.set_index('timestamp')

    dfs_mean.plot()
    plt.show()

    # from sklearn.metrics import r2_score
    # r2 = r2_score(list(prediction_and_actual['P_pool_historical']), list(prediction_and_actual['predicted']))
    pass


def compare_prediction_results():
    dfs = {'[\'minute_15\', \'day_of_week\']_non_static': pd.read_csv(
        '../prediction_results/prediction_[\'minute_15\', \'day_of_week\']_non_static.csv', delimiter='|'),
        '[\'T_historical\', \'day_of_week\', \'minute_15\']': pd.read_csv(
            '../prediction_results/prediction_[\'T_historical\', \'day_of_week\', \'minute_15\'].csv', delimiter='|'),
        '[\'T_historical\', \'day_of_week\', \'minute_15\']_non_static': pd.read_csv(
            '../prediction_results/prediction_[\'T_historical\', \'day_of_week\', \'minute_15\']_non_static.csv',
            delimiter='|'), '[\'T_historical\', \'day_of_week\']_non_static': pd.read_csv(
            '../prediction_results/prediction_[\'T_historical\', \'day_of_week\']_non_static.csv', delimiter='|'),
        '[\'T_historical\', \'minute_15\']_non_static': pd.read_csv(
            '../prediction_results/prediction_[\'T_historical\', \'minute_15\']_non_static.csv', delimiter='|'),
        '[\'T_historical_train\', \'day_of_week\', \'minute_15\']': pd.read_csv(
            '../prediction_results/prediction_[\'T_historical_train\', \'day_of_week\', \'minute_15\'].csv',
            delimiter='|'), '[\'T_historical_train\', \'minute_15\', \'day_of_week\']_non_static': pd.read_csv(
            '../prediction_results/prediction_[\'T_historical_train\', \'minute_15\', \'day_of_week\']_non_static.csv',
            delimiter='|')}
    eval_df = {}
    for k in dfs.keys():
        eval_df[k] = list(dfs[k]['predicted'])
    eval_df['timestamp'] = dfs['[\'minute_15\', \'day_of_week\']_non_static']['timestamp']
    eval_df['P_pool_historical'] = dfs['[\'minute_15\', \'day_of_week\']_non_static']['P_pool_historical']
    ts = pd.DataFrame(eval_df).set_index('timestamp')

    ts.plot()
    plt.show()
    pass


if __name__ == '__main__':
    # compare_prediction_results()

    L = Loader()
    data = L.get_pool_and_temperature_training().copy().set_index('timestamp')
    time_series = data.copy()

    features = ['T_historical', 'minute_15', 'day_of_week']
    # time_series['P_pool_historical'] = time_series['P_pool_historical'].rolling(4).mean().round(-1)
    time_series['T_historical'] = time_series['T_historical'].rolling(2).mean().round(-1)
    time_series = time_series[pd.to_datetime('2015-01-02T00:00:00'):]

    time_series_valid = time_series[pd.to_datetime('2022-03-20T00:15:00'):]
    time_series_train = time_series[:pd.to_datetime('2022-03-20T00:00:00')]
    # create and train a random forest regressor model
    ran_for_reg = create_and_fit_model(x_train=time_series_train[features],
                                       y_train=time_series_train['P_pool_historical'])
    result = predict(regressor=ran_for_reg,
                     x_pred=time_series_valid[features])

    total = data[['P_pool_historical']][pd.to_datetime('2015-01-02T00:00:00'):]
    res = list(data[pd.to_datetime('2015-01-02T00:00:00'):pd.to_datetime('2022-03-20T00:00:00')]['P_pool_historical'])
    r = list(result)
    for e in r:
        res.append(e)
    total['predicted'] = res

    total[['P_pool_historical', 'predicted']].plot()
    plt.show()

    df = time_series[['P_pool_historical']]
    df['predicted'] = total['predicted']# + time_series['P_pool_historical_seasonal']
    # df[pd.to_datetime('2022-03-20T00:15:00'):].plot()
    # plt.show()

    model_evaluation(df[pd.to_datetime('2022-03-20T00:15:00'):])

    # # STORE STORE STORE
    # try:
    #     df.to_csv(f'../prediction_results/prediction_{features}_non_static.csv',
    #               sep='|')
    #     print('store dataframe was successful')
    # except:
    #     print('store dataframe failed')
    pass
