import pandas as pd


class Loader:

    def __init__(self):
        self.__pool_data = pd.read_csv('../measured_values/pool_2015_2022_handled_outliers.csv', delimiter='|')
        self.__substations_data = pd.read_csv('../measured_values/substations_2015_2022.csv', delimiter='|')
        self.__temperature_data = pd.read_csv('../measured_values/pool_2015_2022.csv', delimiter='|')[['timestamp', 'T_historical']]
        self.__pool_data['timestamp'] = pd.to_datetime(self.__pool_data['timestamp'])
        self.__substations_data['timestamp'] = pd.to_datetime(self.__substations_data['timestamp'])
        self.__temperature_data['timestamp'] = pd.to_datetime(self.__temperature_data['timestamp'])

    def get_data(self):
        return self.__pool_data, self.__substations_data, self.__temperature_data
