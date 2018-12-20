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
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
from self_organizing_map import self_organize_map , compare_node_with_weight_matrix,calculate_distance
from copy import deepcopy

def select_closest(candidates, origin):
    """Return the index of the closest candidate to a given point."""
    return euclidean_distance(candidates, origin).argmin()

def euclidean_distance(a, b):
    """Return the array of distances of two numpy arrays of points."""
    return np.linalg.norm(a - b, axis=1)

def get_route(cities, network):
    """
    
    For each city, calculate the neuron closest resembling
    it. say it is n1. find the index of n1 in the weight matrix
    .
    the route will cities , sorted by the order of appearance
    of their matching neuron n1 in the weight matrix
    
    """
    
    route = {}
    count = 0
    for city in cities:

        index =compare_node_with_weight_matrix(city, network)

        route[count] = index 

        count+=1

    
    return sorted(route, key = lambda x: route[x][0])
    # cities['winner'] = cities.apply(
    #     lambda c: select_closest(network, c),
    #     axis=1, raw=True)

    # return cities.sort_values('winner').index



FILE_NAME = 'tsp_data_set/dj38.tsp.txt'



data = panda.read_csv(FILE_NAME, delimiter=' ',skiprows=10)

header = data.columns.values

cities = []

cities.append([float(header[1]), float(header[2])])

data.columns = ['id','x','y']
for i in range(len(data)):
    cities.append([data.iloc[i]['x'], data.iloc[i]['y']])

number_of_cities = len(cities)

original_city = deepcopy(cities)
cities = MinMaxScaler(feature_range= (0,1)).fit_transform(cities)

# 2. create a network and feed to SOM

epoch = 2000
weight_shape = (number_of_cities * 8, 1)
learning_rate = 0.5
tau = 1000
sigma = 12 ## initial radius will be half way of the weight matrix dimension (5x,5y) /2 = 2.5

weight_matrix = self_organize_map(cities, weight_shape,epoch,learning_rate,tau,sigma, iteration_max= 800)

print(weight_matrix)

route = get_route(cities, weight_matrix)

print(route, len(route), len(set(route)))

adjacency_matrix = []

for count in range(number_of_cities):

    row = []

    for inner_count in range(number_of_cities):
        row.append(calculate_distance(np.reshape(original_city[count],2), np.reshape(original_city[inner_count],2)))

    adjacency_matrix.append(row)

sum = 0
for i in range(len(route) -2) :
    sum = sum + adjacency_matrix[i][i+1]

print(sum)