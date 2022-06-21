# === Creates and trains classification models ===

from collections import Counter
import glob
import numpy as np
import tensorflow as tf
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import callbacks
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tslearn.metrics import dtw
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import sys
import pickle
pd.options.mode.chained_assignment = None
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import functions
import classifier_class

def get_files():
    """**Returns files from a specified directory**
    """
    return glob.glob('../data/*')


def load_file(file, type):
    """**Reads one interval from a .csv file**
    Parameters:

    1. **file** - (str) path to file
    2. **type** - (str) type of classifier

    Returns:

    1. **array** - (np array) read interval
    """
    df = pd.read_csv(open(file, 'r'), header=0, sep=';', usecols=['time_ms', 'accelaration_aX_g', 'accelaration_aY_g',
                                                                  'accelaration_aZ_g', 'gyroscope_aX_mdps',
                                                                  'gyroscope_aY_mdps', 'gyroscope_aZ_mdps',
                                                                  'magnetometer_aX_mT', 'magnetometer_aY_mT',
                                                                  'magnetometer_aZ_mT'])
    N = int(df.shape[0] / 2)
    df = df.iloc[N:]
    df = df.reset_index(drop=True)
    if type == 'k':
        df, _ = functions.resample(df, 4)
    elif (type == 's'):
        df, _ = functions.resample(df, 5)
    else:
        df, _ = functions.resample(df, 4)
    if df.empty:
        print('Wrong format of file: ', file)
        return None
    return df.to_numpy()


def flatten(x):
    """**Flattens a 3D np array**
    Parameters:

    1. **x** - (np array) array to be flattened

    Returns:

    1. **flatX** - (np array) flattened array
    """
    flatX = np.empty((x.shape[0], x.shape[2]))
    for i in range(x.shape[0]):
        flatX[i] = x[i, x.shape[1] - 1, :]
    return flatX


def prepare_data(files, type, i=1):
    """**Reads and prepares data from files**
    Parameters:

    1. **files** - (list) list of files
    2. **type** - (str) type of classifier
    3. **i** - (int) can be set to change random_state of train_test_split

    Returns:

    1. **trainX** - (np array) training dataset features
    2. **trainY** - (np array) training dataset labels
    3. **testX** - (np array) testing dataset features
    4. **testY** - (np array) testing dataset labels
    """
    loaded = list()
    labels = list()
    label_classes = tf.constant(['Move_1', 'Move_2', 'Move_3', 'Move_4', 'Move_5', 'Move_6', 'Move_7', 'Move_8', 'Move_9'])
    for file in files:
        data = load_file(file, type)
        if data is None:
            continue
        loaded.append(data)
        pattern = tf.constant(eval('file[8:14]'))
        for i in range(len(label_classes)):
            if re.match(pattern.numpy(), label_classes[i].numpy()):
                labels.append(i)
    loaded = np.asarray(loaded)
    labels = np.asarray(labels).astype('float32')
    labels = tf.keras.utils.to_categorical(labels)
    trainX, testX, trainY, testY = train_test_split(loaded, labels, test_size=0.4, random_state=i*25)   
    return trainX, trainY, testX, testY


def supervised_lstm(trainX, trainY, valX, valY):
    """**Creates a BLSTM supervised neural network model**
    Parameters:

    1. **trainX** - (np array) training dataset features
    2. **trainY** - (np array) training dataset labels
    3. **valX** - (np array) validation dataset features
    4. **valY** - (np array) validation dataset labels

    Returns:

    1. **model** - BLSTM NN
    2. **batch_size** - batch size of the NN
    """
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]

    model = Sequential()
    model.add(Bidirectional(LSTM(units=100, input_shape=(n_timesteps, n_features), return_sequences=True)))
    model.add(Bidirectional(LSTM(units=100, input_shape=(n_timesteps, n_features))))
    model.add(Dropout(0.5))
    model.add(Dense(units=100, activation='tanh'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    earlystopping = callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=20, restore_best_weights=True)
    verbose, epochs, batch_size = 0, 150, 32
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(valX, valY), callbacks=[earlystopping])
    return model, batch_size


def unsupervised_lstm_autoencoder(trainX, trainY, valX, valY):
    """**Creates a LSTM AE and then modifies it for classification**
    Parameters:

    1. **trainX** - (np array) training dataset features
    2. **trainY** - (np array) training dataset labels
    3. **valX** - (np array) validation dataset features
    4. **valY** - (np array) validation dataset labels

    Returns:

    1. **model2** - LSTM AE usable for classification
    2. **batch_size**  - batch size of the NN
    """
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]

    model = Sequential()
    model.add(LSTM(units=150, input_shape=(n_timesteps, n_features), return_sequences=True))
    model.add(LSTM(units=150, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(RepeatVector(n_timesteps))
    model.add(LSTM(units=150, return_sequences=True))
    model.add(LSTM(units=150, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(loss='mae', optimizer='adam')

    earlystopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
    verbose, epochs, batch_size = 0, 500, 32
    model.fit(trainX, trainX, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(valX, valX), callbacks=[earlystopping])

    model2 = Sequential()
    model2.add(LSTM(units=150, input_shape=(n_timesteps, n_features), return_sequences=True, weights=model.layers[0].get_weights()))
    model2.add(LSTM(units=150, input_shape=(n_timesteps, n_features), weights=model.layers[1].get_weights()))
    model2.add(Dense(n_outputs, activation='softmax'))
    model2.layers[0].trainable = False
    model2.layers[1].trainable = False
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    earlystopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
    verbose, epochs, batch_size = 0, 500, 32
    history = model2.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(valX, valY), callbacks=[earlystopping])
    return model2, batch_size


def kmeoids_dtw(d2_trainX):
    """**Builds a kmedoids classificator**
    Parameters:

    1. **d2_trainX** - (np array) flattened 2D training dataset features

    Returns:

    1. **model** - classification model
    2. **cluster_amount** - (int) amount of clusters
    """
    model = KMedoids(n_clusters=8, metric=dtw, random_state=10)
    
    visualizer = KElbowVisualizer(model, k=(3,11), metric='distortion', timings=False)
    visualizer.fit(d2_trainX)
    cluster_amount = visualizer.elbow_value_
    model = KMedoids(n_clusters=cluster_amount, metric=dtw, random_state=10)
    model.fit(d2_trainX)
    return model, cluster_amount


def build_model(type, trainX, trainY, testX, testY, test=True):
    """**Builds a specified model and then either tests it or saves it**
    Parameters:

    1. **type** - (str) type of model to be used
    2. **trainX** - (np array) training dataset features
    3. **trainY** - (np array) training dataset labels
    4. **testX** - (np array) testing dataset features
    5. **testY** - (np array) testing dataset labels
    6. **test** - (bool) true if test is to be done

    Returns:

    1. **accuracy** - (float) accuracy of the classification
    """
    scaler = StandardScaler().fit(flatten(trainX))
    trainX = functions.scale(trainX, scaler)
    testX = functions.scale(testX, scaler)
    if not test:
        pickle.dump(scaler, open('scaler_pkl', 'wb'))
    if not test:
        trainX = np.concatenate((trainX, testX))
        trainY = np.concatenate((trainY, testY))
        trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.3)
    else:
        testX, valX, testY, valY = train_test_split(testX, testY, test_size=0.25)
    
    accuracy = 0

    if type == 's':
        model, batch_size = supervised_lstm(trainX, trainY, valX, valY)
        if test:
            _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
        else:
            classifier = classifier_class.Classifier(model, 'snn', 5)
            pickle.dump(classifier, open('nns_model_pkl', 'wb'))

    if type == 'u':
        model, batch_size = unsupervised_lstm_autoencoder(trainX, trainY, valX, valY)
        if test:
            _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
        else:
            classifier = classifier_class.Classifier(model, 'unn', 4)
            pickle.dump(classifier, open('nnu_model_pkl', 'wb'))

    if type == 'k':
        samples, x , y = trainX.shape
        d2_trainX = trainX.reshape((samples, x * y))
        samples, x , y = testX.shape        
        d2_testX = testX.reshape((samples, x * y))

        pca = PCA(.80)
        d2_trainX = pca.fit_transform(d2_trainX)
        d2_testX = pca.transform(d2_testX)

        model, cluster_amount = kmeoids_dtw(d2_trainX)

        cluster_distribution = {i: [] for i in range(cluster_amount)}
        cluster_labels = []
        predY = model.predict(d2_testX)
        roundedY = np.argmax(testY, axis=1)
        for i in range(len(predY)):
            cluster_distribution[predY[i]].append(roundedY[i])
        if test:
            fig, axes = plt.subplots(cluster_amount)
        for i in range(cluster_amount):
            counted = Counter(cluster_distribution[i])
            cluster_labels.append(counted.most_common(1)[0][0])
            if test:
                key_list = list(counted.keys())
                val_list = list(counted.values())
                axes[i].pie(val_list, labels=key_list, startangle=90, radius=1800)
                axes[i].set_title('Distribution of cluster ' + str(i + 1), fontsize=12)
                axes[i].axis('equal')
        correct = 0
        false = 0
        for i in range(len(predY)):
            true_label = cluster_labels[predY[i]]
            if true_label != roundedY[i]:
                false = false + 1
            else:
                correct = correct + 1
        accuracy = correct / (correct + false)
        if test:
            plt.show()
        else:
            classifier = classifier_class.Classifier(model, 'kmedoids', 5, cluster_labels, pca)
            pickle.dump(classifier, open('km_model_pkl', 'wb'))

    return accuracy


def result(scores):
    """**Prepares and prints statistics of the classification**
    """
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def train(repeats=20):
    """**Trains a specified model which is either saved of tested**
    Parameters:

    1. **repeats** - (int) says how many times the model should be tested
    """
    scores = list()
    if len(sys.argv) == 2:
        dataset = get_files()
        trainX, trainY, testX, testY = prepare_data(dataset, str(sys.argv[1]))
        _ = build_model(str(sys.argv[1]), trainX, trainY, testX, testY, False)
    elif len(sys.argv) == 3 and str(sys.argv[2]) == 't': 
        dataset = get_files()
        trainX, trainY, testX, testY = prepare_data(dataset, str(sys.argv[1]))
        for i in tf.range(repeats):
            score = build_model(str(sys.argv[1]), trainX, trainY, testX, testY)
            score = score * 100.0
            print('>#%d: % .3f' % (i + 1, score))
            scores.append(score)
            trainX, trainY, testX, testY = prepare_data(dataset, str(sys.argv[1]), i + 1)
        result(scores)
    else:
        print('Wrong arguments passed. First argument: "s" (supervised NN) "u" (unsupervised NN) "k" (kmeans)' +
        'or "m" (mix of kmeans and NN). Second argument: Pass "t" to initiate testing of accuracy or no args to train and save the model.')


if __name__ == '__main__':
    train()
