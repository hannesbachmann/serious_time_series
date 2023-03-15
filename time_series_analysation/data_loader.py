import pandas as pd


class Loader:

    def __init__(self):
        self.__pool_data = pd.read_csv('../measured_values/pool_2015_2022.csv', delimiter='|')
        self.__substations_data = pd.read_csv('../measured_values/substations_2015_2022.csv', delimiter='|')

    def get_data(self):
        return self.__pool_data, self.__substations_data
