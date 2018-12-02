import  numpy as np 
import pandas as panda  
from matplotlib import pyplot as plot
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score,precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.callbacks import Callback

pickled_form = 'C:\\Users\\somak\\Documents\\somak_python\\real_world_use_cases_1\\supervised\\classification\\deep_learning\\artificial_neuron_network\\mnist\\minst_compressed.npz'

def load_from_pickled_form(path = pickled_form):

    data =  np.load(path)

    return data.get('x_train'), data.get('x_test'), data.get('y_train'), data.get('y_test')

def prepare_classifier(x, y):

    shape_of_input = x.shape
    shape_of_target = y.shape
    
    classifier  = Sequential()

    ## number of neurons = 30
    ## kernel_initializer determines how the weights are initialized
    ## activation is the activation function at this particular hidden layer
    classifier.add(Dense(units = 30, activation = 'relu',kernel_initializer = 'uniform', input_dim = shape_of_input[1]))

    classifier.add(Dense(units = 30, activation = 'relu',kernel_initializer = 'uniform'))

    classifier.add(Dense(10, activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier


def fit(classifier, x_train, y_train, epoch_size):
    classifier.fit(x_train, y_train, batch_size = 10, epochs = epoch_size )

def predict(classifier, x_test, y_test):

    y_predicted = classifier.predict(x_test)

    return y_predicted

def plot_metrics(y_data , epoch_size, xlabel, ylabel, title):

    plot.plot(list(range(1, epoch_size + 1)), y_data)

    plot.xlabel(xlabel)

    plot.ylabel(ylabel)

    plot.title(title)

    plot.show()

def execute():
    
    x_train, x_test, y_train, y_test = load_from_pickled_form()

    print('shape of input ', x_train.shape)
    print('shape of target ', y_train.shape)

    epoch_size = 100
    classifier = prepare_classifier(x_train[:15000], y_train[:15000])
    # fit(classifier, x_train[:15000], y_train[:15000], epoch_size )

    # y_predicted = predict(classifier, x_test, y_test)


execute()


