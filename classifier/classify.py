import numpy as np
import tensorflow as tf
import pandas as pd
import sys
from threading import Thread, Event
from queue import Queue
from os import path, getcwd
from scipy.spatial import distance
THIS_DIR = getcwd()
MODBUS_API_DIR = path.abspath(path.join(THIS_DIR, 'modbus_api'))
DEVICE_API_DIR = path.abspath(path.join(THIS_DIR, 'device_api'))
sys.path.append(MODBUS_API_DIR)
sys.path.append(DEVICE_API_DIR)
import device_send
import pickle
pd.options.mode.chained_assignment = None


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


class Classifier:
    def __init__(self, model, type, cluster_labels = None):
        self.model = model
        self. type = type
        self.cluster_labels = cluster_labels

    def print_single(self, label, i):
        print('>#%d: Possibility of labels: ' % i)
        for l in label:
            print('% .2f%%' % (l * 100))              
        print(' Predicted label: %d' % (np.argmax(label) + 1))

    def print_multiple(self, labels, i):
        for label in labels:
            self.print_single(label, i)          
            i = i + 1
    
    def predict_2D(self, data, i):
        if (self.type != 'kmedoids'):
            label = self.model.predict(np.array([data,]))[0]
            self.print_single(label, i)
        else:
            data = np.expand_dims(data, axis=0)
            _, x, y = data.shape
            data_2D = data.reshape((1, x * y))
            label = self.model.predict(data_2D)
            label = self.cluster_labels[label[0]]
            print('>#%d: Predicted label: %d' % (i, label + 1))
    
    def predict_3D(self, data, i):
        if (self.type != 'kmedoids'):
            labels = self.model.predict(data)
            self.print_multiple(labels, i)
        else:
            samples, x , y = data.shape
            data_2D = data.reshape((samples, x * y))
            labels = self.model.predict(data_2D)
            print(labels)
            labels = [self.cluster_labels[i] for i in labels]
            print(labels)
            for label in labels:
                print('>#%d: Predicted label: %d' % (i, label + 1))
                i = i + 1

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
    df_tmp = df_tmp.resample('1000ms').mean()
    df_tmp.index = pd.RangeIndex(start=0, stop=3, step=1)
    df_tmp.drop('time_ms', inplace=True, axis=1)
    return df_tmp, index


def scale(x, scaler):
    for i in range(x.shape[0]):
        x[i, :, :] = scaler.transform(x[i, :, :])
    return x


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
                df_tmp, index = resample(df_tmp)
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
        tmp, index = resample(df)
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
        data = scale(data, scaler)
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
            classifier = Classifier(model, 'snn')
        elif x == 'u':
            model = pickle.load(open('nnu_model_pkl', 'rb'))
            scaler = pickle.load(open('scaler_pkl', 'rb'))
            classifier = Classifier(model, 'unn')
        elif x == 'k':
            model = pickle.load(open('km_model_pkl', 'rb'))
            scaler = pickle.load(open('scaler_pkl', 'rb'))
            cluster_labels = pickle.load(open('km_labels_pkl', 'rb'))
            classifier = Classifier(model, 'kmedoids', cluster_labels)
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