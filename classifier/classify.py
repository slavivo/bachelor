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
import classifier_class
import pickle
pd.options.mode.chained_assignment = None

def real_time_eval(classifier, scaler):
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
                'gyroscope_aY_mdps', 'gyroscope_aZ_mdps', 'magnetometer_aX_mT', 'magnetometer_aY_mT', 'magnetometer_aZ_mT']]
                df_tmp, index = functions.resample(df_tmp)
                if df_tmp.empty:
                    print('Wrong format of input data from sensor.')
                    continue
                data = df_tmp.to_numpy()
                if scaler != None:
                    data = scaler.transform(data)
                classifier.predict_2D(data, counter)
                counter += 1
                df = df.iloc[150:]
        except KeyboardInterrupt:
            event.set()
            t.join()
            exit()

def load_file(file):
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
        tmp, index = functions.resample(df)
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
    data = load_file(input_file)
    if scaler != None:
        data = functions.scale(data, scaler)
    classifier.predict_3D(data, 1)

def wrong_args():
    print('Wrong arguments passed. First argument: "s" (supervised NN) "u" (unsupervised NN) or "k" (kmedoids). Second argument: can be a file name with input to classify.')

def classify():
    input_file = None
    scaler = None 
    if len(sys.argv) > 1:
        x = str(sys.argv[1])
        if x == 's':
            model = pickle.load(open('nns_model_pkl', 'rb'))
            classifier = classifier_class.Classifier(model, 'snn')
        elif x == 'u':
            model = pickle.load(open('nnu_model_pkl', 'rb'))
            scaler = pickle.load(open('scaler_pkl', 'rb'))
            classifier = classifier_class.Classifier(model, 'unn')
        elif x == 'k':
            model = pickle.load(open('km_model_pkl', 'rb'))
            scaler = pickle.load(open('scaler_pkl', 'rb'))
            cluster_labels = pickle.load(open('km_labels_pkl', 'rb'))
            classifier = classifier_class.Classifier(model, 'kmedoids', cluster_labels)
        else:
            wrong_args()
            return
        if len(sys.argv) > 2:
            input_file = str(sys.argv[2])
    else:
        wrong_args()
        return
    if input_file == None:
        real_time_eval(classifier, scaler)
    else:
        eval_from_file(classifier, input_file, scaler)

if __name__ == '__main__':
    classify()