import pandas as pd
import feather
import matplotlib.pyplot as plt


class DataSink:
    def __init__(self):
        pass

    def tmp_store_data(self, df, name):
        feather.write_dataframe(df, '../evaluation/tmp_datasets/' + name)

    def tmp_store_plot(self, fig, name):
        fig.savefig('../evaluation/tmp_plots/' + name + '.png')

    def store_preprocessed_data(self, df, name):
        feather.write_dataframe(df, '../evaluation/preprocessed_data/' + name)

    def store_analysis_plot(self, fig, name):
        fig.savefig('../evaluation/analysis_plots/' + name + '.png')

    def store_prediction_result_data(self, df, name):
        feather.write_dataframe(df, '../evaluation/prediction_results_data/' + name)

    def store_prediction_result_plot(self, fig, name):
        fig.savefig('../evaluation/prediction_results_plots/' + name + '.png')

    def store_model(self, model, name):
        pass

    def tmp_store_model(self, model, name):
        pass
