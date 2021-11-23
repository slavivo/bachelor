import glob
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pynput


def get_files():
    return glob.glob("../data/*")


def load_file(file):
    df = pd.read_csv(open(file, 'r'), nrows=375, header=0, sep=';', names=["accelaration_aX_g", "accelaration_aY_g",
                                                              "accelaration_aZ_g", "gyroscope_aX_mdps",
                                                              "gyroscope_aY_mdps", "gyroscope_aZ_mdps"])
    if df.shape[0] != 375:
        print("Wrong format of file: ", file)
        return None
    return df.values


def prepare_data(files):
    loaded = list()
    labels = list()
    label_classes = tf.constant(["Move_1", "Move_2", "Move_3", "Move_4", "Move_5"])
    for file in files:
        data = load_file(file)
        if data is None:
            continue
        loaded.append(data)
        pattern = tf.constant(eval("file[8:14]"))
        for i in range(len(label_classes)):
            if re.match(pattern.numpy(), label_classes[i].numpy()):
                labels.append(i)
    loaded = np.asarray(loaded)
    print("Labels before categorical transformation: ", labels)
    labels = np.asarray(labels).astype('float32')
    labels = tf.keras.utils.to_categorical(labels)
    print("Dataset shape: ", loaded.shape, "Labels shape: ", labels.shape)
    trainX, testX = np.array_split(loaded, 2)
    trainY, testY = np.array_split(labels, 2)
    print("Number of sample in training dataset: ", len(trainX), " In testing dataset: ", len(testX))
    return trainX, trainY, testX, testY


def eval_model(trainX, trainY, testX, testY):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    model = tf.keras.models.Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    verbose, epochs, batch_size = 0, 15, 64
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
    return accuracy


def result(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def classify(repeats=5):
    dataset = get_files()
    trainX, trainY, testX, testY = prepare_data(dataset)
    scores = list()
    for i in range(repeats):
        score = eval_model(trainX, trainY, testX, testY)
        score = score * 100.0
        print('>#%d: % .3f' % (i + 1, score))
        scores.append(score)
    result(scores)



if __name__ == '__main__':
    classify()


