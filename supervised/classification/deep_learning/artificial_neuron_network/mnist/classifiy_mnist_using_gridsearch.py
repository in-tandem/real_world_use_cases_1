import  numpy as np
import pandas as panda
from matplotlib import pyplot as plot
import keras 
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score,precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from keras.callbacks import Callback
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from collections import Counter

pickled_form = 'C:\\Users\\somak\\Documents\\somak_python\\real_world_use_cases_1\\supervised\\classification\\deep_learning\\artificial_neuron_network\\mnist\\minst_compressed.npz'

def load_from_pickled_form(path = pickled_form):

    data =  np.load(path)

    return data.get('x_train'), data.get('x_test'), data.get('y_train'), data.get('y_test')

def prepare_classifier(x_data, y_data):
    
    shape_of_input = x_data.shape
    shape_of_target = y_data.shape

    classifier  = Sequential()

    ## number of neurons = 30
    ## kernel_initializer determines how the weights are initialized
    ## activation is the activation function at this particular hidden layer
    ## input_shape is the number of features in a single row.. in this case it is shape_of_input[1]
    ## shape_of_input[0] is the total number of such rows
    classifier.add(Dense(units = 30, activation = 'relu', kernel_initializer = 'uniform', input_dim = shape_of_input[1]))

    classifier.add(Dense(units = 30, activation = 'relu', kernel_initializer = 'uniform'))

    ## we are predicting 10 digits for each row of x.
    ## in total there are shape_of_input[0] rows in total
    classifier.add(Dense(10, activation = 'softmax'))

    ## categorical_crossentropy is the loss function for multi output loss function
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return classifier


def fit(classifier, x_train, y_train, epoch_size, batch_size = 10):


    pipeline = Pipeline([
                ('keras_classifier', classifier)
        ])

    param_grid = {

        'keras_classifier__batch_size' : [10,20,30,50],
        'keras_classifier__epochs' : [100, 200, 300],
        'keras_classifier__x_data' : [x_train],
        'keras_classifier__y_data' : [y_train],
        
    }


    grid = GridSearchCV(estimator = pipeline, param_grid = param_grid, n_jobs = -1)
    grid.fit(x_train, y_train)

    print("Best parameters are : ", grid.best_params_, '\n grid best score :', grid.best_score_)


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

    ## lets scale these input values . pixels ranges are between 0 -255. so dividing by 255 will give us a range of 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    epoch_size = 100
    ## output variables i.e y values are already between 0 - 9.
    ## y values are between 0 -9. this is a multiclass classification
    ## problem. we would perform a one hot encoding to the target 
    ## variable using keras.to_Categorical matrix.
    ## this will in essence convert each to a binary matrix
    ## our final loss function calculator will be crossentropy for
    ## multiclass classification. keras additionally dictates 
    ## that using categorical_crossentropy as a loss function
    ## required to_categorical subset
    ## the below shows how to convert back

    '''
        >>> a = np.array([1,0,2,2,3,5])
        >>> to_categorical(a)
        array([[0., 1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1.]], dtype=float32)
        >>> bb=to_categorical(a)
        >>> np.argmax(bb, axis = 1)
        array([1, 0, 2, 2, 3, 5], dtype=int64)
    '''
    y_train = keras.utils.to_categorical(y_train[:100], 10)
    y_test_categorized = keras.utils.to_categorical(y_test, 10)

    classifier =  KerasClassifier(build_fn = prepare_classifier)

    fit(classifier, x_train[:100], y_train, epoch_size )

'''
    y_predicted = predict(classifier, x_test, y_test_categorized)

    ##reconverting categorical to actual values
    y_predicted_arg_max = np.argmax(y_predicted, axis = 1)
    print(y_predicted_arg_max, '\n unique values are :',np.unique(y_predicted_arg_max))


    diff = np.argmax(y_test_categorized, axis =1) - y_predicted_arg_max

    counter = Counter(diff)   

    print(counter)
    correct_guesses = counter.get(0)
    print('Total number of values correctly guessed - ', correct_guesses, \
            '\n Incorrect guesses - ', y_predicted_arg_max.shape[0] - correct_guesses)


    accuracies_across_epochs = classifier.history.history.get('acc')

    print('testing acc ', classifier.history.history.get('val_acc'))
    plot_metrics(accuracies_across_epochs , epoch_size, 'Epochs', 'Accuracy', 'Accuracy across epoch')

''' 

execute()


