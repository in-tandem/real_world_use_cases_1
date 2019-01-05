import numpy as np 
import pandas as panda 
from sklearn.model_selection import KFold
from sklearn.datasets import make_regression
from sklearn.base import clone


def generate_regression_data(n_rows = 100, n_cols = 5):

    return make_regression(n_samples = n_rows, n_features = n_cols, random_state = 12)


def split_data_into_hold_out():
    pass

## get a model trained with outliers
## get a model trained without outliers
## predict average of each


## stack a couple of good regression models on elo data and check score

class StackedRegressor(object):
    
    '''
    for each model:

        divide up the train data into n_splits fold.
        use one for training the model
        use the trained model on other folds for making predictions
        use the trained model for predicting on the test set


    '''
    def __init__(self,base_model, meta_model, n_splits):
                
        self.base_model = base_model
        self.meta_model = meta_model
        self.splits = n_splits
        


    def fit(self, x, y, test_x):
        """

        The method will split the given data set into n_splits given in the init method

        1. For each model given, it will train on one split, and predict on the rest
        2. For each model given, it will train on one split, and predict on test data

        The given dataset out of step 1 will serve as train data for meta model
        The given dataset out of step 2 will serve as test data for meta model

        :param x: dataframe
        :param y : data frame
        """
        
        count = 0 

        total_size = len(x)
        folds = np.random.randint(1, self.splits +1, (total_size))
        x['fold_id'] = folds
        y['fold_id'] = folds

        test_fold_size = len(np.where(folds!=1)[0])

        x_train = x[x['fold_id'] == 1]
        y_train = y[y['fold_id'] == 1][[i for i in y.columns.tolist() if i!='fold_id']]

        x_test = x[x['fold_id'] != 1]
        y_test = y[y['fold_id'] != 1][[i for i in y.columns.tolist() if i!='fold_id']]

        ## num of base models + 1. +1 to have the test values for y
        self.meta_train_data = panda.DataFrame(np.zeros((len(test_fold_size), len(self.base_model) + 1)))
        self.meta_test_data = np.zeros((len(test_x), len(self.base_model)))
        self.meta_train_data[self.meta_train_data.shape[1]] = y_test

        for k, model in enumerate(self.base_model):
            
            model.fit(x_train,y_train)
            y_prediction = model.predict(x_test)
            self.meta_train_data[k] = y_prediction
            self.meta_test_data[k] = model.predict(test_x)
    
    def predict(self):
        ## TODO whats the y value for training. from where do i get it        
        self.meta_model.fit(self.meta_train_data.loc[: self.meta_train_data.shape[0],:\
                    self.meta_train_data.shape[1] - 2], \
                self.meta_train_data[self.meta_train_data.shape[1]-1])
        
        return self.meta_model.predict(self.meta_test_data)

class AverageRegressor(object):
    
    def predict(self, y_predictions):
        """

        :param y_predictions: sequence of y_predictions in np.array format

        """
        return np.average(y_predictions, axis = 0)


