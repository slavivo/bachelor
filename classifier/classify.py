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
pd.options.mode.chained_assignment = None

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
    df_tmp = df_tmp.resample('100ms').mean()
    df_tmp.index = pd.RangeIndex(start=0, stop=30, step=1)
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
                df = df.iloc[50:]
        except KeyboardInterrupt:
            event.set()
            t.join()
            exit()

def load_file(file):
    loaded = list()
    try:
        df = pd.read_csv(open(file, 'r'), header=0, sep=';', usecols=['time_ms', 'accelaration_aX_g', 'accelaration_aY_g',
                                                                      'accelaration_aZ_g', 'gyroscope_aX_mdps',
                                                                      'gyroscope_aY_mdps', 'gyroscope_aZ_mdps'])
    except FileNotFoundError:
        print('File does not exist')
        return
    while(True):
        tmp, index = resample(df)
        if tmp.empty:
            break
        loaded.append(tmp.values)
        df = df.iloc[index:]
        df = df.reset_index(drop=True)
    if len(loaded) == 0:
        print('Wrong format of file')
        return
    loaded = np.asarray(loaded)
    return loaded
    

def eval_from_file(model, input_file):
    data = load_file(input_file)
    labels = model.predict(data)
    counter = 1
    for label in labels:
        print('>#%d: Possibility of labels: ' % counter)
        for l in label:
            print('% .2f%%' % (l * 100))
        print(' Predicted label: %d' % (np.argmax(label) + 1))

def wrong_args():
    print('Wrong arguments passed. First argument: "s" (supervised NN) "u" (unsupervised NN) "k" (kmeans) or "m" (mix of kmeans and NN). Second argument: can be a file name with input to classify.')

def classify():
    input_file = None 
    if len(sys.argv) > 1:
        x = str(sys.argv[1])
        if x == 's':
            model = pickle.load(open('nns_model_pkl', 'rb'))
        elif x == 'u':
            model = pickle.load(open('nnu_model_pkl', 'rb'))
        elif x == 'k':
            model = pickle.load(open('k_model_pkl', 'rb'))
        elif x == 'm':
            model = pickle.load(open('m_model_pkl', 'rb'))
        else:
            wrong_args()
            return
        if len(sys.argv) > 2:
            input_file = str(sys.argv[2])
    else:
        wrong_args()
        return
    if input_file == None:
        real_time_eval(model)
    else:
        eval_from_file(model, input_file)

if __name__ == '__main__':
    classify()