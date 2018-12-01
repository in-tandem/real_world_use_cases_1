import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
XX = dataset.iloc[:, 3:13].values
yy = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
XX[:, 1] = labelencoder_X_1.fit_transform(XX[:, 1])
labelencoder_X_2 = LabelEncoder()
XX[:, 2] = labelencoder_X_2.fit_transform(XX[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
tt = onehotencoder.fit_transform(XX).toarray()
#X = X[:, 1:]# -*- coding: utf-8 -*-

