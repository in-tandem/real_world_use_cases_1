
import pandas as panda
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
from self_organizing_map import self_organize_map , compare_node_with_weight_matrix,calculate_distance
from copy import deepcopy
from multiprocessing import Pool 
import random
from minisom import MiniSom

FILE_NAME = 'tsp_data_set/dj38.tsp.txt'
def get_route(cities, network):
    """
    
    For each city, calculate the neuron closest resembling
    it. say it is n1. find the index of n1 in the weight matrix
    .
    the route will be cities , sorted by the order of appearance
    of their matching neuron n1 in the weight matrix
    
    """
    route = []

    for city in cities:

        route.append(compare_node_with_weight_matrix(city, network))

    print('foudn route: ', route)    
    return sorted(route)




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


# input_data = scale_input_data(create_input_data(198000))
minisom = MiniSom(cities.shape[0]*20,1, 2,sigma=0.1,learning_rate=0.12)
minisom.random_weights_init(cities)
minisom.train_random(cities,1)
network = minisom.get_weights()
# print(network, network.shape)
route= get_route(cities,network)
print(route, len(route), len(set(route)))

adjacency_matrix = []
final_route=[]
for count in range(number_of_cities):

    row = []

    for inner_count in range(number_of_cities):
        row.append(calculate_distance(np.reshape(original_city[count],2), np.reshape(original_city[inner_count],2)))
        final_route.append(original_city[inner_count])
    adjacency_matrix.append(row)

sum = 0

for i in range(len(route) -2) :
    sum = sum + adjacency_matrix[i][i+1]

print(sum, final_route)