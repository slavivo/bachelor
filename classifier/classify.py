import numpy as np
import tensorflow as tf
import pandas as pd
import sys
from threading import Thread, Event
from queue import Queue
from os import path, getcwd
THIS_DIR = getcwd()
MODBUS_API_DIR = path.abspath(path.join(THIS_DIR, 'modbus_api'))
DEVICE_API_DIR = path.abspath(path.join(THIS_DIR, 'device_api'))
sys.path.append(MODBUS_API_DIR)
sys.path.append(DEVICE_API_DIR)
import device_send
import pickle

def resample(df):
    # Resampling to 33hz for now
    time = df['time_ms']
    start = time[0]
    end = start + 10 * 300  # 300 * 10 ms -> 3s
    index = None
    for i in range(10):  # Find closest time value
        index = time.where(time == end).first_valid_index()
        if index:
            break
        end += 1
    if not index:
        return pd.DataFrame(), 0
    df_tmp = df.head(index)
    df_tmp['time_delta'] = pd.to_timedelta(df_tmp['time_ms'], 'ms')
    df_tmp.index = df_tmp['time_delta']
    df_tmp = df_tmp.resample('10ms').mean()
    df_tmp.index = pd.RangeIndex(start=0, stop=300, step=1)
    df_tmp.drop('time_ms', inplace=True, axis=1)
    return df_tmp, index

def real_time_eval(model):
    counter = 1
    df = pd.DataFrame()
    queue = Queue()
    event = Event()
    t = Thread(target=device_send.main, args=(queue, event))
    t.start()
    print('\n---Beginning real-time classfication---\n')
    while(True):
        try:
            if not queue.empty():
                df_tmp = queue.get()
                df = df.append(df_tmp, ignore_index=True)
                # print('Queue size:', queue.qsize())
                # print('---Main thread df---\n', df.head(2), df.tail(2), '\n Shape:', df.shape[0])
            if event.is_set() and queue.empty():
                break
            if df.shape[0] > 350:
                df_tmp = df[['time_ms', 'accelaration_aX_g', 'accelaration_aY_g', 'accelaration_aZ_g', 'gyroscope_aX_mdps',
                    'gyroscope_aY_mdps', 'gyroscope_aZ_mdps']]
                df_tmp, index = resample(df_tmp)
                if df_tmp.empty:
                    print('Wrong format of input data from sensor.')
                    continue
                data = df_tmp.values
                label = model.predict(np.array([data,]))[0]
                print('>#%d: Possibility of labels: ' % counter)
                for l in label:
                    print('% .2f%%' % (l * 100))
                print(' Predicted label: %d' % (np.argmax(label) + 1))
                counter += 1
                df = df.iloc[index:]
        except KeyboardInterrupt:
            event.set()
            t.join()
            exit()

def classify():
    model = pickle.load(open('nn_model_pkl', 'rb'))
    real_time_eval(model)

if __name__ == '__main__':
    classify()