from collections import Counter
import glob
import numpy as np
import tensorflow as tf
import pandas as pd
import re
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import callbacks
from scipy.spatial import distance
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
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
    df_tmp = df_tmp.resample('500ms').mean()
    df_tmp.index = pd.RangeIndex(start=0, stop=6, step=1)
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


def flatten(x):
    flatX = np.empty((x.shape[0], x.shape[2]))
    for i in range(x.shape[0]):
        flatX[i] = x[i, x.shape[1] - 1, :]
    return flatX


def scale(x, scaler):
    for i in range(x.shape[0]):
        x[i, :, :] = scaler.transform(x[i, :, :])
    return x


def prepare_data(files):
    loaded = list()
    labels = list()
    label_classes = tf.constant(['Move_1', 'Move_2', 'Move_3', 'Move_4', 'Move_5', 'Move_6'])
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
    trainX, testX, trainY, testY = train_test_split(loaded, labels, test_size=0.5, random_state=40)   
    return trainX, trainY, testX, testY

def DTW(a, b):
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0
    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost
    return cumdist[an, bn]


def supervised_lstm(trainX, trainY, valX, valY):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]

    model = Sequential()
    model.add(Bidirectional(LSTM(100, input_shape=(n_timesteps, n_features))))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    earlystopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    verbose, epochs, batch_size = 1, 50, 32
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(valX, valY), callbacks=[earlystopping])
    return model, batch_size


def unsupervised_lstm_autoencoder(trainX, trainY, valX, valY):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]

    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(RepeatVector(n_timesteps))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

    earlystopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    verbose, epochs, batch_size = 0, 500, 32
    model.fit(trainX, trainX, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(valX, valX), callbacks=[earlystopping])

    model_output = Dense(n_outputs, activation='relu')(model.layers[0].output)
    model = Model(inputs=model.inputs, outputs=model_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    earlystopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    verbose, epochs, batch_size = 0, 500, 32
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(valX, valY), callbacks=[earlystopping])
    return model, batch_size


def eval_model(type, trainX, trainY, testX, testY, test=True):
    if type != 's':
        scaler = StandardScaler().fit(flatten(trainX))
        trainX = scale(trainX, scaler)
        testX = scale(testX, scaler)
    if not test:
        trainX = np.concatenate((trainX, testX))
        trainY = np.concatenate((trainY, testY))
        trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.25)
    else:
        testX, valX, testY, valY = train_test_split(testX, testY, test_size=0.25)
        print('Number of sample in training dataset: ', len(trainX), ' In validation dataset: ', len(valX), ' In testing dataset: ', len(testX))
    
    accuracy = 0

    if type == 's':
        model, batch_size = supervised_lstm(trainX, trainY, valX, valY)
        if test:
            _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
        else:
            pickle.dump(model, open('nns_model_pkl', 'wb'))

    if type == 'u':
        model, batch_size = unsupervised_lstm_autoencoder(trainX, trainY, valX, valY)
        if test:
            _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
        else:
            pickle.dump(model, open('nns_model_pkl', 'wb'))

    if type == 'k':
        model = KMedoids(n_clusters=6, metric=DTW)
        samples, x , y = trainX.shape
        d2_trainX = trainX.reshape((samples, x * y))
        samples, x , y = testX.shape        
        d2_testX = testX.reshape((samples, x * y))
        # visualizer = KElbowVisualizer(model, k=(1,10))
        # visualizer.fit(d2_trainX)
        # visualizer.show()
        model.fit(d2_trainX)
        cluster_distribution = {i: [] for i in range(6)}
        if test:
            predY = model.predict(d2_testX)
            roundedY = np.argmax(testY, axis=1)
            print(predY)
            print(roundedY)
            for i in range(len(predY)):
                cluster_distribution[predY[i]].append(roundedY[i])
            print(cluster_distribution)
            fig, axes = plt.subplots(6)
            # fig.suptitle('Clouster label distribution')
            for i in range(6):
                counted = Counter(cluster_distribution[i])
                key_list = list(counted.keys())
                val_list = list(counted.values())
                axes[i].pie(val_list, labels=key_list, startangle=90, radius=1800)
                axes[i].set_title('Distribution of cluster ' + str(i + 1), fontsize=12)
                axes[i].axis('equal')
            plt.show()
        else:
            pickle.dump(model, open('nns_model_pkl', 'wb'))

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
    if len(sys.argv) == 2:
        _ = eval_model(str(sys.argv[1]), trainX, trainY, testX, testY, False)
    elif len(sys.argv) == 3 and str(sys.argv[2]) == 't':
        for i in tf.range(repeats):
            score = eval_model(str(sys.argv[1]), trainX, trainY, testX, testY)
            score = score * 100.0
            print('>#%d: % .3f' % (i + 1, score))
            scores.append(score)
        result(scores)
    else:
        print('Wrong arguments passed. First argument: "s" (supervised NN) "u" (unsupervised NN) "k" (kmeans)' +
        'or "m" (mix of kmeans and NN). Second argument: Pass "t" to initiate testing of accuracy or no args to train and save the model.')


if __name__ == '__main__':
    train()
