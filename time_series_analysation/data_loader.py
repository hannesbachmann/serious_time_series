import pandas as pd


class Loader:

    def __init__(self):
        self.__pool_data = pd.read_csv('../measured_values/pool_2015_2022_handled_outliers.csv', delimiter='|')
        self.__substations_data = pd.read_csv('../measured_values/substations_2015_2022.csv', delimiter='|')
        self.__temperature_data = pd.read_csv('../measured_values/pool_2015_2022.csv', delimiter='|')[['timestamp', 'T_historical']]
        self.__p_and_t = pd.read_csv('../measured_values/pool_temperature_2015_2022_handled_outliers.csv', delimiter='|')
        self.__pool_data['timestamp'] = pd.to_datetime(self.__pool_data['timestamp'])
        self.__substations_data['timestamp'] = pd.to_datetime(self.__substations_data['timestamp'])
        self.__temperature_data['timestamp'] = pd.to_datetime(self.__temperature_data['timestamp'])
        self.__p_and_t['timestamp'] = pd.to_datetime(self.__p_and_t['timestamp'])

    def get_data(self):
        return self.__pool_data, self.__substations_data, self.__temperature_data

    def get_pool_and_temperature(self):
        return self.__p_and_t
