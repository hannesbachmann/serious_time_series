import feather
import pandas as pd


class DataLoader:
    def __init__(self):
        pass

    def load_all(self):
        return feather.read_dataframe('../measured_values/pool_2015_2022')

    def load_handled_outliers(self):
        return feather.read_dataframe('../measured_values/pool_2015_2022_handled_outliers')

    def load_handled_outliers_static(self):
        return feather.read_dataframe('../measured_values/pool_2015_2022_handled_outliers_static')

    def load_temperature_handled_outliers(self):
        return feather.read_dataframe('../measured_values/pool_temperature_handled_outliers')

    def load_training(self):
        return feather.read_dataframe('../measured_values/pool_2015_2022_training')


class DataConverter:
    def __init__(self):
        pass

    def convert_csv_to_feather(self, name):
        df = pd.read_csv(name, delimiter='|')
        name = name.replace('.csv', '')
        feather.write_dataframe(df, name)
        pass

    def convert_feather_to_csv(self, name):
        df = feather.read_dataframe(name)
        df.to_csv(name + '.csv')
        pass


if __name__ == '__main__':
    # D = DataConverter()
    # D.convert_csv_to_feather(name='../measured_values/substations_2015_2022.csv')
    L = DataLoader()
    L.load_training()
