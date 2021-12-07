import glob
import numpy as np
import tensorflow as tf
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
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
pd.options.mode.chained_assignment = None

def get_files():
    return glob.glob('../data/*')


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
    df_tmp = df_tmp.resample('30ms').mean()
    df_tmp.index = pd.RangeIndex(start=0, stop=100, step=1)
    df_tmp.drop('time_ms', inplace=True, axis=1)
    return df_tmp, index


def load_file(file):
    df = pd.read_csv(open(file, 'r'), header=0, sep=';', usecols=['time_ms', 'accelaration_aX_g', 'accelaration_aY_g',
                                                                  'accelaration_aZ_g', 'gyroscope_aX_mdps',
                                                                  'gyroscope_aY_mdps', 'gyroscope_aZ_mdps'])
    df, _ = resample(df)
    if df.empty:
        print('Wrong format of file: ', file)
        return None
    return df.values


def prepare_data(files):
    loaded = list()
    labels = list()
    label_classes = tf.constant(['Move_1', 'Move_2', 'Move_3', 'Move_4', 'Move_5'])
    label_classes = tf.constant(["Move_1", "Move_2", "Move_3", "Move_4", "Move_5"])
    for file in files:
        data = load_file(file)
        if data is None:
            continue
        loaded.append(data)
        pattern = tf.constant(eval('file[8:14]'))
        for i in range(len(label_classes)):
            if re.match(pattern.numpy(), label_classes[i].numpy()):
                labels.append(i)
    loaded = np.asarray(loaded)
    print('Labels before categorical transformation: ', labels)
    labels = np.asarray(labels).astype('float32')
    labels = tf.keras.utils.to_categorical(labels)
    print('Dataset shape: ', loaded.shape, 'Labels shape: ', labels.shape)
    trainX, testX = np.array_split(loaded, 2)
    trainY, testY = np.array_split(labels, 2)
    print('Number of sample in training dataset: ', len(trainX), ' In testing dataset: ', len(testX))
    return trainX, trainY, testX, testY


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

def eval_model(trainX, trainY, testX, testY, test=True):
    if not test:
        trainX = np.concatenate((trainX, testX))
        trainY = np.concatenate((trainY, testY))
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    accuracy = 0
    model = tf.keras.models.Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    verbose, epochs, batch_size = 0, 15, 64
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    if test:
        _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
    else:
        real_time_eval(model)
    return accuracy


def result(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def classify(repeats=5):
    # load_file('../data/Move_1_001.csv')
    dataset = get_files()
    trainX, trainY, testX, testY = prepare_data(dataset)
    scores = list()
    if len(sys.argv) == 1:
        for i in tf.range(repeats):
            score = eval_model(trainX, trainY, testX, testY)
            score = score * 100.0
            print('>#%d: % .3f' % (i + 1, score))
            scores.append(score)
        result(scores)
    elif str(sys.argv[1]) == 'rt':
        _ = eval_model(trainX, trainY, testX, testY, False)
    else:
        print('Wrong arguments passed. Pass "rt" to initiate real-time classification.')


if __name__ == '__main__':
    classify()


