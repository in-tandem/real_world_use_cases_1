from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import f1_score,auc, roc_auc_score,roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelBinarizer, label_binarize
import numpy as np 
from matplotlib import pyplot as plot 
from sklearn.tree import DecisionTreeClassifier
from itertools import cycle
from scipy import interp
from multi_class_plot_utils import calculate_roc_and_prec_metrics, \
        plot_roc_auc_curve,plot_precision_recall_curve

x,y = load_iris(return_X_y= True)

# y = label_binarize(y, classes = np.unique(y))

x_train,x_test,y_train, y_test = train_test_split(x,y, stratify = y, test_size = 0.3)

classifier = [
    OneVsRestClassifier(estimator = DecisionTreeClassifier()),
    DecisionTreeClassifier()
]

classifier_name = [
    'one_v_rest',
    'dt'
]
classifier_param_grid = [

    {
        'one_v_rest__estimator__max_depth' : [2,4,6],
        'one_v_rest__estimator__criterion' : ['gini']
    },
    {
        'dt__max_depth' : [2,4,6],
        'dt__criterion' : ['gini']
    }
]

for model, model_name, model_param in zip(classifier, classifier_name, classifier_param_grid):

    pipeline = Pipeline(

        [
            ('scaler', StandardScaler()),
            (model_name, model)
        ]

    )

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=12)

    gridsearch = GridSearchCV(estimator = pipeline, param_grid = model_param, cv = cv, n_jobs = -1, scoring = 'accuracy')

    search = gridsearch.fit(x_train,y_train)

    y_prediction = search.predict(x_test)

    # classes = np.unique(y_train)

    if hasattr(gridsearch.best_estimator_, 'predict_proba'):
            
        print('inside decision function')
        y_prob = gridsearch.predict_proba(x_test)
        
        number_of_classes = len(np.unique(y_train))
        y_test_bin = label_binarize(y_test, classes = np.unique(y_train))
        response = calculate_roc_and_prec_metrics(y_test_bin, y_prob, number_of_classes = number_of_classes)

        roc_params = {

            'false_positive_rate_across_class': response.get('false_positive_rate_across_class'),
            'true_positive_rate_across_class': response.get('true_positive_rate_across_class'),
            'roc_auc_across_class': response.get('roc_auc_across_class'),
            'n_classes': number_of_classes
        }

        precision_recall_params  = {

            'precision_across_class': response.get('precision_across_class'),
            'average_precision_across_class': response.get('average_precision_across_class'),
            'recall_across_class': response.get('recall_across_class'),
            'n_classes': number_of_classes
        }

        plot_roc_auc_curve(**roc_params)
        plot_precision_recall_curve(**precision_recall_params)

