import pandas as panda
import numpy as np
import datetime, time
from matplotlib.pyplot import plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import seaborn as sns
from sklearn.linear_model import LinearRegression, RANSACRegressor, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from xgboost import XGBRegressor
from math import sqrt

## we will split outliers in an 80-20 split in training and testing data
## we will totally ignore outliers
## we will take average of the above two predictions


classifiers = [
    GradientBoostingRegressor(loss='huber'),
]


classifier_names = [
            'gbr',               
    
]

classifier_param_grid = [
            {'gbr__n_estimators' :[1000]},
    #         {'boost_regressor__max_depth':[3,5,6],\
    #              'boost_regressor__learning_rate':[0.1,0.05], \
    #                 'boost_regressor__reg_alpha':[0.1,0.2,0.3], \
    #                     'boost_regressor__reg_lambda':[1,2,3,0.5,0.6], \
				# 		'boost_regressor__n_estimators':[100,200,250]
				# 		},
    
]


def root_mean_square_error(y, y_predicted):
    
    return mean_squared_error(y,y_predicted)

scorer = make_scorer(root_mean_square_error, greater_is_better=False)


class CodeTimer:
    
    """
        Utility custom contextual class for calculating the time 
        taken for a certain code block to execute
    
    """
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''

    def __enter__(self):
        self.start = time.clock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (time.clock() - self.start) * 1000.0
        time_taken = datetime.timedelta(milliseconds = self.took)
        print('Code block' + self.name + ' took(HH:MM:SS): ' + str(time_taken))



def runGridSearchAndPredict(pipeline, x_train, y_train, x_test, y_test, param_grid, n_jobs = 1, cv = 10, score = 'neg_mean_squared_error'):
    
    response = {}
    training_timer       = CodeTimer('training')
    testing_timer        = CodeTimer('testing')

    with training_timer:

        gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, cv = cv, n_jobs = n_jobs, scoring = score)

        search = gridsearch.fit(x_train,y_train)

        print("Grid Search Best parameters ", search.best_params_)
        print("Grid Search Best score ", search.best_score_)
            
    with testing_timer:
        y_prediction = gridsearch.predict(x_test)
            
    print("Mean squared error score %s" %mean_squared_error(y_test,y_prediction))
    
    response['testing_time'] = testing_timer.took
    response['_y_prediction'] = y_prediction
    response['training_time'] = training_timer.took    
    response['mean_squared_error'] = mean_squared_error(y_test,y_prediction)
    response['root_mean_squared_error'] = search.best_score_
    response['r2_score'] = r2_score(y_test,y_prediction)
    response['best_estimator'] = search.best_estimator_
    
    return response
    


def analyzeRegressionModel(X,y, outliers):

    
    _x_train, _x_test, _y_train, _y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)

    ## simply ignoring all outliers
    outlier_x = outliers[[i for i in outliers.columns.tolist() if i not in ['target','card_id']]]
    outlier_y =  outliers[['target']]
    
    outlier_x_train, outlier_x_test,outlier_y_train, outlier_y_test = \
        train_test_split(outlier_x, outlier_y, test_size = 0.2, random_state = 2)

    print('x train shape:', len(_x_train),' , y train shape:', len(_y_train))
    print('x test shape:', len(_x_test),' , y test shape:', len(_y_test))
    _x_train = panda.concat([_x_train, outlier_x_train])
    
    _y_train = panda.concat([_y_train, outlier_y_train])
    
    _x_test = panda.concat([_x_test, outlier_x_test])
    
    _y_test = panda.concat([_y_test, outlier_y_test])
    
    model_metrics = {}

    for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):

            pipeline = Pipeline([
                    ('scaler', RobustScaler()),
                    (model_name, model)
            ])

            cross_validator = KFold(n_splits = 10, random_state = 12)
            result = runGridSearchAndPredict(pipeline, _x_train, _y_train, _x_test, _y_test, model_param_grid, cv = cross_validator,score = scorer)

#             result = runGridSearchAndPredict(pipeline, _x_train, _y_train, _x_test, _y_test, model_param_grid,score = scorer)

            _y_prediction = result['_y_prediction']

            model_metrics[model_name] = {}
            model_metrics[model_name]['training_time'] = result['training_time']
            model_metrics[model_name]['testing_time'] = result['testing_time']
            model_metrics[model_name]['r2_score'] = result['r2_score']
            model_metrics[model_name]['mean_squared_error'] = result['mean_squared_error']
            model_metrics[model_name]['root_mean_squared_error'] = result['root_mean_squared_error']
            model_metrics[model_name]['best_estimator'] = result['best_estimator']
            
    return model_metrics
    print('Model metrics are \n :', model_metrics)
	
def execute():
    
    train_data_path = 'all/train_new_details_added.csv'
    test_data_path = 'all/test_new_details_added.csv'
    
    train_data = panda.read_csv(train_data_path)
    test_data = panda.read_csv(test_data_path)
    
    train_data = train_data[[i for i in train_data.columns.tolist() if i not in ['Unnamed: 0']]]

    without_outliers = train_data[train_data['target'] > -29]
    with_outliers = train_data[train_data['target'] < -29]
    
    round1_x = without_outliers[[i for i in without_outliers.columns.tolist() if i not in ['target','card_id']]]
    
    round1_y = without_outliers[['target']]
            
    metrics = analyzeRegressionModel(round1_x, round1_y, with_outliers)	
    
    # best_estimator =  metrics['gbr']['best_estimator']
    
    # test_data = test_data[['feature_1','feature_2','feature_3','months_passed','total_no_of_transaction','total','total_accepted', \
    # 'total_rejects','mean_installments','max_cat_1','max_cat_2','cat_3','most_appearing_mc_id','mean_month_lag']]
    
    # test_target = best_estimator.predict(test_data)
    
    
    # temp = panda.read_csv('all/test.csv')
    # sample_submission_round_1 = temp[['card_id']]
    # sample_submission_round_1['target'] = test_target
    
    print(metrics)
    
    # print(len(sample_submission_round_1), sample_submission_round_1.head())
    
    # sample_submission_round_1.to_csv('round_xgboost_new_details_submission.csv')

execute()