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
import pickle
pd.options.mode.chained_assignment = None
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

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
    df_tmp = df_tmp.resample('10ms').mean()
    df_tmp.index = pd.RangeIndex(start=0, stop=300, step=1)
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
        pickle.dump(model, open('nn_model_pkl', 'wb'))
    return accuracy


def result(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def train(repeats=5):
    # load_file('../data/Move_1_001.csv')
    dataset = get_files()
    trainX, trainY, testX, testY = prepare_data(dataset)
    scores = list()
    if len(sys.argv) == 1:
        _ = eval_model(trainX, trainY, testX, testY, False)
    elif str(sys.argv[1]) == 't':
        for i in tf.range(repeats):
            score = eval_model(trainX, trainY, testX, testY)
            score = score * 100.0
            print('>#%d: % .3f' % (i + 1, score))
            scores.append(score)
        result(scores)
    else:
        print('Wrong arguments passed. Pass "t" to initiate testing of accuracy or no args to train and save the model.')


if __name__ == '__main__':
    train()
