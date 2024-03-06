"""power prediction model using a random forest regressor"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from time_series_analysation.data_loader import Loader
from tensorflow import keras
import sklearn as sk
import statsmodels
import seaborn
from data_source_sink.data_source import DataLoader


def create_and_fit_random_forest_model(x_train, y_train):
    """x_train: temperature, minute, day_of_week, holiday
    y_train: power
    temperature and power are static"""
    regressor = RandomForestRegressor(n_estimators=20,
                                      max_depth=25,
                                      random_state=42)
    regressor.fit(x_train, y_train)
    return regressor


def create_and_fit_lstm_model(x_train, y_train, x_test, y_test):
    """x_train: temperature, minute, day_of_week, holiday
    y_train: power
    temperature and power are static"""
    x_train.columns = [0, 1, 2, 3]
    x_train = x_train[[i for i in range(4)]].values
    y_train = pd.DataFrame(y_train).reset_index(drop=True)
    y_train.columns = [0]

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=120,
        sampling_rate=6,
        batch_size=256,
    )

    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)

    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    path_checkpoint = "model_checkpoint.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    dataset_val = create_validation_dataset(x_test, y_test)

    history = model.fit(
        dataset_train,
        epochs=10,
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback],
    )

    visualize_loss(history, "Training and Validation Loss")

    for x, y in dataset_val.take(5):
        show_plot(
            [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],
            12,
            "Single Step Prediction",
        )

    return dataset_train


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    return


def predict(regressor, x_pred):
    y_pred = regressor.predict(x_pred)
    return y_pred


def create_validation_dataset(x_test, y_test):
    x_test.columns = [0, 1, 2, 3]
    x_test = x_test[[i for i in range(4)]].values
    y_train = pd.DataFrame(y_test).reset_index(drop=True)
    y_train.columns = [0]

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_test,
        y_test,
        sequence_length=120,
        sampling_rate=6,
        batch_size=256,
    )
    return dataset_val


def model_evaluation(prediction_and_actual):
    # calculate mean error for each day
    prediction_and_actual['err'] = abs(prediction_and_actual['predicted'] - prediction_and_actual['P_pool_historical'])
    dfs = [g for n, g in prediction_and_actual[['err']].groupby(pd.Grouper(freq='H'))]
    # create dataframe to store the means for every step based on the given frequency
    dfs_mean = pd.DataFrame({'timestamp': [g.index[0] for g in dfs],
                             f'err': [g['err'].mean() for g in dfs]})
    dfs_mean['err_cum_sum'] = dfs_mean['err'].cumsum()
    dfs_mean['counter'] = [c for c in range(1, dfs_mean.shape[0] + 1)]
    dfs_mean['err_cum_mean'] = dfs_mean['err_cum_sum'] / dfs_mean['counter']
    dfs_mean = dfs_mean.set_index('timestamp')

    dfs_mean[['err_cum_mean', 'err']].plot()
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


def prepare_prediction(time_series, features):
    data = time_series.copy()
    # time_series['P_pool_historical'] = time_series['P_pool_historical'].rolling(4).mean().round(-1)
    # time_series['T_historical'] = time_series['T_historical'].rolling(2).mean()     # .round(-1)
    # correlate the previous power value with the current power value
    time_series['previous_value'] = [0] + list(time_series['P_pool_historical_train'][:-1])
    time_series = time_series[pd.to_datetime('2015-01-02T00:00:00'):]

    time_series_valid = time_series[pd.to_datetime('2022-03-20T00:15:00'):]
    time_series_train = time_series[:pd.to_datetime('2022-03-20T00:00:00')]
    # create and train a random forest regressor model
    # lstm_history = create_and_fit_lstm_model(x_train=time_series_train[features],
    #                                         y_train=time_series_train['P_pool_historical'],
    #                                         x_test=time_series_valid[features],
    #                                         y_test=time_series_valid['P_pool_historical'])
    # create and train a lstm regressor model
    ran_for_reg = create_and_fit_lstm_model(x_train=time_series_train[features],
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
    df['predicted'] = total['predicted']  # + time_series['P_pool_historical_seasonal']
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


if __name__ == '__main__':
    # compare_prediction_results()

    L = DataLoader()
    data = L.load_training().copy().set_index('timestamp')
    time_series = data.copy()
    features = ['T_historical', 'minute_15', 'day_of_week']
    prepare_prediction(time_series=time_series, features=features)

    pass
