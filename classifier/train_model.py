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
from sklearn.metrics.cluster import completeness_score
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
import functions

def get_files():
    return glob.glob('../data/*')


def load_file(file):
    df = pd.read_csv(open(file, 'r'), header=0, sep=';', usecols=['time_ms', 'accelaration_aX_g', 'accelaration_aY_g',
                                                                  'accelaration_aZ_g', 'gyroscope_aX_mdps',
                                                                  'gyroscope_aY_mdps', 'gyroscope_aZ_mdps',
                                                                  'magnetometer_aX_mT', 'magnetometer_aY_mT',
                                                                  'magnetometer_aZ_mT'])
    N = int(df.shape[0] / 2)
    df = df.iloc[N:]
    df = df.reset_index(drop=True)
    df, _ = functions.resample(df)
    if df.empty:
        print('Wrong format of file: ', file)
        return None
    return df.to_numpy()


def flatten(x):
    flatX = np.empty((x.shape[0], x.shape[2]))
    for i in range(x.shape[0]):
        flatX[i] = x[i, x.shape[1] - 1, :]
    return flatX


def prepare_data(files):
    loaded = list()
    labels = list()
    label_classes = tf.constant(['Move_1', 'Move_2', 'Move_3', 'Move_4', 'Move_5', 'Move_6', 'Move_7', 'Move_8', 'Move_9'])
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


def supervised_lstm(trainX, trainY, valX, valY):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]

    model = Sequential()
    model.add(Bidirectional(LSTM(100, input_shape=(n_timesteps, n_features))))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    earlystopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    verbose, epochs, batch_size = 0, 150, 32
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
    model.compile(loss='mae', optimizer='adam')

    earlystopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    verbose, epochs, batch_size = 0, 500, 32
    model.fit(trainX, trainX, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(valX, valX), callbacks=[earlystopping])

    model_output = Dense(n_outputs, activation='softmax')(model.layers[0].output)
    model = Model(inputs=model.inputs, outputs=model_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    earlystopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    verbose, epochs, batch_size = 0, 500, 32
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(valX, valY), callbacks=[earlystopping])
    return model, batch_size


def eval_model(type, trainX, trainY, testX, testY, test=True):
    if type != 's':
        scaler = StandardScaler().fit(flatten(trainX))
        trainX = functions.scale(trainX, scaler)
        testX = functions.scale(testX, scaler)
        if not test:
            pickle.dump(scaler, open('scaler_pkl', 'wb'))
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
            pred = model.predict(testX, batch_size=batch_size, verbose=0)
            tmp1 = testY.argmax(axis=1) + 1
            tmp2 = pred.argmax(axis=1) + 1
            for i in range(len(tmp1)):
                if tmp1[i] != tmp2[i]:
                    print(str(tmp1[i]) + ' - ' + str(tmp2[i]))
            _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
        else:
            pickle.dump(model, open('nns_model_pkl', 'wb'))

    if type == 'u':
        model, batch_size = unsupervised_lstm_autoencoder(trainX, trainY, valX, valY)
        if test:
            _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
        else:
            pickle.dump(model, open('nnu_model_pkl', 'wb'))

    if type == 'k':
        cluster_amount = 9
        model = KMedoids(n_clusters=cluster_amount, metric=functions.DTW)
        samples, x , y = trainX.shape
        d2_trainX = trainX.reshape((samples, x * y))
        samples, x , y = testX.shape        
        d2_testX = testX.reshape((samples, x * y))
        # visualizer = KElbowVisualizer(model, k=(1,10))
        # visualizer.fit(d2_trainX)
        # visualizer.show()
        model.fit(d2_trainX)
        cluster_distribution = {i: [] for i in range(cluster_amount)}
        cluster_labels = []
        predY = model.predict(d2_testX)
        roundedY = np.argmax(testY, axis=1)
        for i in range(len(predY)):
            cluster_distribution[predY[i]].append(roundedY[i])
        if test:
            print(cluster_distribution)
            fig, axes = plt.subplots(cluster_amount)
            print('Completeness score: %.3f' % completeness_score(roundedY, predY))
            print(predY)
            print(roundedY)
        # fig.suptitle('Clouster label distribution')
        for i in range(cluster_amount):
            counted = Counter(cluster_distribution[i])
            cluster_labels.append(counted.most_common(1)[0][0])
            if test:
                key_list = list(counted.keys())
                val_list = list(counted.values())
                axes[i].pie(val_list, labels=key_list, startangle=90, radius=1800)
                axes[i].set_title('Distribution of cluster ' + str(i + 1), fontsize=12)
                axes[i].axis('equal')
        print(cluster_labels)
        if test:
            plt.show()
        else:
            pickle.dump(model, open('km_model_pkl', 'wb'))
            pickle.dump(cluster_labels, open('km_labels_pkl', 'wb'))

    return accuracy


def result(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def train(repeats=10):
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
