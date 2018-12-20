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
from travel_using_ga import Population
from sklearn.preprocessing import MinMaxScaler
import numpy as np 

FILE_NAME = 'all/cities.csv'
NORMALIZED_X_Y_PATH = 'normalized_kaggle_x_y.npz'

def pickle_to_drive(x, path = NORMALIZED_X_Y_PATH):
    
    np.savez_compressed(path, cities = x)


class Location(object):

    def __init__(self, identifier, x, y):
        self.identifier = identifier
        self.x = x
        self.y = y

def calculate_distance(a, b):
    
    return round(sqrt(  (a.x - b.x)**2 + (a.y - b.y)**2  ))

data = panda.read_csv(FILE_NAME)

header = data.columns.values

cities = []

for i in range(len(data)):
    cities.append([data.iloc[i]['X'], data.iloc[i]['Y']])

number_of_cities = len(cities)

adjacency_matrix= []

cities = MinMaxScaler(feature_range= (0,1)).fit_transform(cities)

pickle_to_drive( x = cities)
