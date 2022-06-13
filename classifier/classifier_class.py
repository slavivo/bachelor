# === Contains classification model ===


import numpy as np
import functions

class Classifier:
    """**A classification model**
    """
    def __init__(self, model, type, cluster_labels = None):
        """**Initializes its type and model**
        Parameters:

        1. **model** - classification model
        2. **type** - (str) specifies which type of a model this is
        3. **cluster_labels** - (list int) tells which cluster corresponds to which label
        """
        self.model = model
        self. type = type
        self.cluster_labels = cluster_labels

    def print_single(self, label, i):
        """**Prints a single prediction**
        Parameters:

        1. **label** - (list float) contains probabilities of each class for the classification
        2. **i** - (int) contains how many real-time classifications were done already
        """
        print('>#%d: Possibility of labels: ' % i)
        for l in label:
            print('% .2f%%' % (l * 100))              
        print(' Predicted label: %d' % (np.argmax(label) + 1))

    def print_multiple(self, labels, i):
        """**Prints multiple predictions**
        Parameters:

        1. **labels** - (list list float) contains a list of probabilities of each class for the classification
        2. **i** - (int) contains how many real-time classifications were done already
        """
        for label in labels:
            self.print_single(label, i)          
            i = i + 1
    
    def predict_2D(self, data, i):
        """**Predicts a single 2D array**
        Parameters:

        1. **data** - (np array) 2D array to be predicted
        2. **i** - (int) contains how many real-time classifications were done already
        """
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
        """**Predicts a single 3D array**
        Parameters:

        1. **data** - (np array) 3D array to be predicted
        2. **i** - (int) contains how many real-time classifications were done already
        """
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