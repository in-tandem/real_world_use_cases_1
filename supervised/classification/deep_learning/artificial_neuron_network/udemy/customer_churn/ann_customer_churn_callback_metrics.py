# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plot
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score,precision_recall_fscore_support
import pandas as panda
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.callbacks import Callback

class CalculateF1ScoreAtEachEpoch(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = {}
        self.val_recalls = {}
        self.val_precisions = {}

    def on_epoch_end(self, epoch, logs={}):
        """
            len(self.validation_data) == 3, because 
            validation_data[0] == train_x (that you input in model.fit()), 
            validation_data[1] == train_y, 
            validation_data[2]=sample_weight,

        validation_data needs to be provided in the fit method
        else self.validation_data will be empty

        """
        val_predict = self.model.predict(self.validation_data[0])
        val_predict = (val_predict > 0.5) ##using the same activation function
        val_targ = self.validation_data[1]
        all_metrics = precision_recall_fscore_support(val_targ, val_predict, average = 'weighted')
        print("val_f1: %f — val_precision: %f — val_recall %f" %(all_metrics[2], all_metrics[0], all_metrics[1]))
        
        self.val_f1s[epoch] = all_metrics[2]
        self.val_recalls[epoch] = all_metrics[1]
        self.val_precisions[epoch] = all_metrics[0]
        return


PATH = "C:\\Users\\somak\\Documents\\somak_python\\real-world-use-cases\\supervised\\classification\\deep_learning\\artificial_neuron_network\\udemy\\customer_churn\\Churn_Modelling.csv"


def get_data(fully_qualified_file_path):
    
    data = panda.read_csv(fully_qualified_file_path)
    
    print('Data found. Size of data is : ', data.shape)
    print('Columns in data: ', data.columns.tolist())
    
    return data

def get_target_variable(data, column_name):
    
    y = data[column_name]
    
#    another way to extract data : y = data.iloc[:, 13].values (data.iloc[:, 13]  is a pandas.core.series.Series object)
    
    print("shape of target: " , y.shape)
    
    print(y.value_counts())
    ## we see there are two unique values and they are numerical. encoding not required
    ## however classes are imbalanced. we would be using our f1 scores for better evaluation of models
    
    return y

def view_prepare_independent_variables(data):
    
    ## initial understanding would be to take all columns except the target columns
    x = data[data.columns.tolist()[:-1]] ## target column is at the end , ignoring it.
    
    
    print("Description of selected independent variables: ", x.info)
    
    ##we can ignore the columns RowNumber	CustomerId	Surname
    ## we would have to encode the data for the columns Geography and Gender
    
    removed_columns = ['RowNumber', 'CustomerId', 'Surname']
    columns = [i for i in x.columns.tolist() if i not in removed_columns]
    
    x = x[columns]
    
    print(x.head(5))
    
    label_encoder = LabelEncoder()
    
    ## can also be done using apply function on pd dataframe
    ## x[2] = x[2].apply(lambda x : 1 if x=='male' else 0)
    
    x['Geography'] = label_encoder.fit_transform(x['Geography'])
    x['Gender'] = label_encoder.fit_transform(x['Gender']) 
    
    print(x.head(), np.unique(x['Geography'].values))
    
    ##in order to prevent falling into dummy variable trap, we would one hot encode
    ## one hot encoding wil return a numpy array, which we can feed into our models laters

    one_hot_encoder = OneHotEncoder(categorical_features=[1]) ## we only want to encode the geography columns

    x = one_hot_encoder.fit_transform(x).toarray()[:, 1:]

    print(x.shape)    
    
    return x

def create_model():
    
    classifier = Sequential()
    
    ##units is the number of neurons in the first hidden layer
    ##activation is the activcation function to be used
    ##input_shape =(11,) is equivalent to input_dim =11, this is numb of features given
    ## this is out first hidden layer
    classifier.add(Dense(units = 6,activation = 'relu',kernel_initializer = 'uniform', input_shape= (11,)))
    
    classifier.add(Dense(units = 6,activation = 'relu',kernel_initializer = 'uniform' ))
    
    
    ##adding our output layer. expecting only one neuron since we have a binary classification
    classifier.add(Dense(1, activation='sigmoid')) ##logistic regression
    
       ##forclassifcation issues, metrics is always accuracy
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    
    return classifier

def plot_accuracy_results(y_data, epoch_size, xlabel, ylabel, title):

    # accuracies_across_epochs = history.get('acc')

    plot.plot(list(range(1, epoch_size + 1)) ,y_data)

    plot.xlabel(xlabel)
    plot.ylabel(ylabel)
    plot.title(title)

    plot.show()



def run():
    """
    
    Steps undertaken would be as follows:
        
        1. Get the data from the remote file
        2. Check the target of the data. Count the distributions. Encode to numerical if necessary
        3. Check the dependent variables. Select the ones that make sense. ENcode,if necessary
        4. train_test_Split
        5. standard scaling
        6. run it through models        
        
    """
    
    data = get_data(PATH)
    
    y = get_target_variable(data = data, column_name = 'Exited')
    
    x = view_prepare_independent_variables(data)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3,stratify = y, shuffle = True, random_state = 1)
    
    print('size of train set: ', x_train.shape)
    print('size of test set: ', x_test.shape)
    print('size of target train set: ', y_train.shape)
    print('size of target test set: ', y_test.shape)
    print('Distribution of discrete values in train target : ', np.bincount(y_train))
    print('Distribution of discrete values in test target : ', np.bincount(y_test))
    
    model = create_model()
    epoch_size = 100 

    ##adding our custom f1 scorer
    custom_metric = CalculateF1ScoreAtEachEpoch()
 
    ##validation_data needs to be provided
    ## else in custom callbacks we would be empty
    model.fit(x_train, y_train, batch_size = 10, epochs = epoch_size, validation_data = (x_train, y_train), callbacks = [custom_metric])
    
    y_predicted = model.predict(x_test)
    
    ## y_predicted so far has only probability values ranging from 0 - 1.
    ## we need to convert it to binary values of 0 or 1
    ## in order to do so we use a threshold function
    ## in our case the htreshold function is 0.5
    ## (y_predicted > 0.5) operates on a numpy array and returns boolean for each item in the array
    y_predicted = (y_predicted > 0.5) 
    
    matrix = confusion_matrix(y_test.values, y_predicted) 
    
    f_score = f1_score(y_test.values, y_predicted)

    ## below calculates at one go the precision, recall and f1 values
    ## average weighted takes care of the unbalanbced target set
    all_metrics = precision_recall_fscore_support(y_test, y_predicted, average = 'weighted')
    
    print(all_metrics)
    print("Classification matrix : ", matrix)
    
    print('F1 score is : ', f_score, " \n precision is :", all_metrics[0], '\n recall is : ', all_metrics[1], ' \n f beta score is: ', all_metrics[2])


    ##we will also print the final accuracy score, evenb though for unbalanced data sets
    ## accuracy scores are rarely right
    
    print('accuracy score is : ', accuracy_score(y_test, y_predicted))

    ## in order to get the accuracies at each epoch
    ## model.history.history.get('acc')
    
    plot_accuracy_results(model.history.history.get('acc'),\
            epoch_size, xlabel = 'Epochs', ylabel = 'Accuracy', \
                title = 'Accuracy across each epoch')

    print('f1 at the end of each epoch ', custom_metric.val_f1s)

    plot_accuracy_results(custom_metric.val_f1s.values(),\
            epoch_size, xlabel = 'Epochs', ylabel = 'Accuracy', \
                title = 'Accuracy across each epoch')
    
    
run()


    
        