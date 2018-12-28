"""

this is using TSP data set from the website http://www.math.uwaterloo.ca/tsp/data/index.html
the sample below uses country Djibouti
According to website optimal path is 6656
Total number of cities is 38
this is a sizeable figure

Our code below will read the tsp.txt file.
Create locations instances out of them and create adjacency matrix as well

"""

import pandas as panda
from math import sqrt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np 

FILE_NAME = 'all/cities.csv'
NORMALIZED_X_Y_PATH = 'normalized_kaggle_x_y.npz'

def pickle_to_drive(x, path = NORMALIZED_X_Y_PATH):
    
    np.savez_compressed(path, cities = x)


data = panda.read_csv(FILE_NAME)


cities = []

for i in range(len(data)):
    cities.append([data.iloc[i]['X'], data.iloc[i]['Y']])

number_of_cities = len(cities)

cities1 = np.round(cities)
cities2 = np.round(MinMaxScaler(feature_range= (0,1)).fit_transform(cities1), decimals = 4)


# cities2 = StandardScaler().fit_transform(cities1)
# cities = np.round()
pickle_to_drive( x = cities2)
