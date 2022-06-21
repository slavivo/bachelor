# === Performs real-time or file classification ===


import numpy as np
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
import functions
import pickle
pd.options.mode.chained_assignment = None

def real_time_eval(classifier, scaler):
    """**Starts real-time classification**
    Parameters:

    1. **classifier** - (Classifier) classification model
    2. **scaler** - scaler usable on 2D data
    """
    counter = 1
    df = pd.DataFrame()
    queue = Queue()
    event = Event()
    t = Thread(target=device_send.main, args=(queue, event))
    duration = classifier.duration
    t.start()
    print('\n---Beginning real-time classfication---\n')
    while(True):
        try:
            if not queue.empty():
                df_tmp = queue.get()
                df = df.append(df_tmp, ignore_index=True)
            if event.is_set() and queue.empty():
                break
            if df.shape[0] > duration * 120:
                df_tmp = df[['time_ms', 'accelaration_aX_g', 'accelaration_aY_g', 'accelaration_aZ_g', 'gyroscope_aX_mdps',
                'gyroscope_aY_mdps', 'gyroscope_aZ_mdps', 'magnetometer_aX_mT', 'magnetometer_aY_mT', 'magnetometer_aZ_mT']]
                df_tmp, index = classifier.resample(df_tmp)
                if df_tmp.empty:
                    print('Wrong format of input data from sensor.')
                    continue
                data = df_tmp.to_numpy()
                if scaler != None:
                    data = scaler.transform(data)
                classifier.predict_2D(data, counter)
                counter += 1
                df = df.iloc[(int) (duration / 2 * 100):]
        except KeyboardInterrupt:
            event.set()
            t.join()
            exit()

def load_file(file, classifier):
    """**Reads whole .csv file into a np array**
    Parameters:

    1. **file** - (str) path to file
    2. **classifier** - (Classifier) classification model

    Returns:

    1. **loaded** - (np array) array of loaded data from file
    """
    loaded = list()
    try:
        df = pd.read_csv(open(file, 'r'), header=0, sep=';', usecols=['time_ms', 'accelaration_aX_g', 'accelaration_aY_g',
                                                                      'accelaration_aZ_g', 'gyroscope_aX_mdps',
                                                                      'gyroscope_aY_mdps', 'gyroscope_aZ_mdps',
                                                                      'magnetometer_aX_mT', 'magnetometer_aY_mT',
                                                                      'magnetometer_aZ_mT'])
    except FileNotFoundError:
        print('File does not exist')
        return
    while(True):
        tmp, index = classifier.resample(df)
        if tmp.empty:
            break
        loaded.append(tmp.to_numpy())
        df = df.iloc[index:]
        df = df.reset_index(drop=True)
    if len(loaded) == 0:
        print('Wrong format of file')
        return
    loaded = np.asarray(loaded)
    return loaded
    

def eval_from_file(classifier, input_file, scaler):
    """**Performs classification on specified file**
    Parameters:

    1. **classifier** - (Classifier) classification model
    2. **input_file** - (str) path to file
    3. **scaler** -  scaler usable on 2D data
    """
    data = load_file(input_file, classifier)
    if scaler != None:
        data = functions.scale(data, scaler)
    classifier.predict_3D(data, 1)

def wrong_args():
    """**Prints message in case of invalid args**
    """
    print('Wrong arguments passed. First argument: "s" (supervised NN) "u" (unsupervised NN) or "k" (kmedoids). Second argument: can be a file name with input to classify.')

def classify():
    """**Performs real-time or file classification**
    """
    input_file = None
    scaler = None 
    if len(sys.argv) > 1:
        x = str(sys.argv[1])
        if x == 's':
            classifier = pickle.load(open('nns_model_pkl', 'rb'))
        elif x == 'u':
            classifier = pickle.load(open('nnu_model_pkl', 'rb'))
        elif x == 'k':
            classifier = pickle.load(open('km_model_pkl', 'rb'))
        else:
            wrong_args()
            return
        if len(sys.argv) > 2:
            input_file = str(sys.argv[2])
    else:
        wrong_args()
        return
    scaler = pickle.load(open('scaler_pkl', 'rb'))
    if input_file == None:
        real_time_eval(classifier, scaler)
    else:
        eval_from_file(classifier, input_file, scaler)

if __name__ == '__main__':
    classify()