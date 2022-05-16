import numpy as np
import functions

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