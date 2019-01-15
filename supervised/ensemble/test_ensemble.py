import numpy as np 
import pandas as panda 
from sklearn.model_selection import KFold, train_test_split
from sklearn.datasets import make_regression
from sklearn.base import clone
from sklearn.linear_model import LinearRegression,SGDRegressor, RANSACRegressor
from stacking_ensemble import StackedRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def generate_regression_data(n_rows = 1000, n_cols = 5):

    return make_regression(n_samples = n_rows, n_features = n_cols, random_state = 12)

base_models = [SGDRegressor(), RANSACRegressor()]

meta_model = LinearRegression()

def execute():
    X,y = generate_regression_data()

    X = panda.DataFrame(X)
    y = panda.DataFrame(y)
    print(X.shape, y.shape)

    _x_train, _x_test, _y_train, _y_test = train_test_split(X,y,test_size = 0.2, random_state =123)
    
    _x_train_1 = _x_train.copy()
    _x_test_1 = _x_test.copy()
    _y_train_1 = _y_train.copy()
    _y_test_1 = _y_test.copy()

    all_my_models = {}
    regressor = StackedRegressor(base_model = base_models, \
                            meta_model = meta_model, \
                            n_splits = 5)

    regressor.fit(_x_train,_y_train,_x_test)

    y_predictions = regressor.predict()
    all_my_models['stacked']= mean_squared_error(_y_test,y_predictions)

    for model in base_models:
            
        model.fit(_x_train_1,_y_train_1)
        y_predictions = model.predict(_x_test_1)
        all_my_models[model.__class__.__name__] = mean_squared_error(_y_test_1, y_predictions)

    ## you will see that on the same test data, stacked ensemble consistently gives lower 
    ## error
    print('mean_abs_error: ', all_my_models)


execute()
