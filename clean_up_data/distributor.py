import threading
from queue import Queue
import pandas as pd
import time
from time_series_analysation.data_loader import Loader
from clean_up_data.handle_outliers import handle_to_high_values
from clean_up_data.handle_outliers import handle_shut_down_values
from clean_up_data.handle_outliers import handle_low_day_values


def preprocessing_pipeline(ts, q):
    tmp_time_series = ts.copy()
    tmp_time_series = handle_to_high_values(ts=tmp_time_series)
    tmp_time_series = handle_shut_down_values(ts=tmp_time_series, replacing_method='values')
    tmp_time_series = handle_low_day_values(ts=tmp_time_series, replacing_method='values')
    q.put(tmp_time_series)


def distribute_clean_ups(ts):
    # yearly distribute
    quarters = [g.reset_index() for n, g in
                ts.set_index('timestamp').groupby(pd.Grouper(freq='Q'))]  # or 'Q' for quarter
    # distributor
    threads = []
    queues = []
    for y in quarters:
        q = Queue()
        thread = threading.Thread(target=preprocessing_pipeline, args=(y, q,))
        thread.start()
        threads.append(thread)
        queues.append(q)
    print('threads started')
    for t in threads:
        t.join()
    print('threads finished')
    quarters_results = []
    for q in queues:
        quarters_results.append(q.get())
    print('queues got')
    df = pd.concat(quarters_results, ignore_index=True)
    # res_dict = {'timestamp': [],
    #             'P_pool_historical': [],
    #             'T_hisctorical': []}
    # for yr_i in range(len(quarters_results)):
    #     res_dict['timestamp'].append(quarters_results[yr_i]['timestamp'])
    return df


if __name__ == '__main__':
    L = Loader()
    pool, substations, temperature = L.get_data()
    time_series = pool[['timestamp', 'P_pool_historical']].copy()
    time_series['T_historical'] = temperature['T_historical']

    time_before = time.time()
    time_series = distribute_clean_ups(ts=time_series)
    print(f'total time needed: {time.time() - time_before}')  # 14.25sec

    # STORE STORE STORE
    try:
        time_series.set_index('timestamp').to_csv('C:\\Users\\Hannes\\PycharmProjects\\serious_time_series\\measured_values\\pool_temperature_2015_2022_handled_outliers.csv', sep='|')
        print('store dataframe was successful')
    except:
        print('store dataframe failed')
