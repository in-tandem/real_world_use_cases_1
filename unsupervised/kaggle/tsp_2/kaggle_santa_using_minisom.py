
import pandas as panda
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
from copy import deepcopy
from multiprocessing import Pool 
import random
from minisom import MiniSom
from scipy.spatial import KDTree

FILE_NAME = 'all/cities.csv'

def compare_node_with_weight_matrix(x, w):

    reshaped_net = np.reshape(w, (w.shape[0],w.shape[2]))
    reshaped_net = [i for i in reshaped_net if np.nan not in i]

    #TODO kdtree is getting NAN.remove all nan
    point, index = KDTree(reshaped_net).query(x)
    print('i found index ', index)
    return index


def calculate_distance(a, b):
    """
    this method would calculate the euclidean distance between
    points a and b.

    essentially this will serve as a distance measure between our
    node and weight neuron.

    """
    # return np.sum((a-b) ** 2)
    return np.linalg.norm(a - b)



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

    print('found route: ', route)

    return sorted(route)

def get_cities():
    
    data = panda.read_csv(FILE_NAME)


    cities = []

    for i in range(len(data)):
        cities.append([data.iloc[i]['X'], data.iloc[i]['Y']])

    return cities

def calculate_final_route():
    pass

def execute():

    cities = get_cities()

    number_of_cities = len(cities)

    original_city = deepcopy(cities)

    # cities = MinMaxScaler(feature_range= (0,1)).fit_transform(cities)
    cities = np.reshape(cities, (len(cities),2))
    minisom = MiniSom(cities.shape[0]*3,1, 2,sigma=22,learning_rate = 5)

    minisom.random_weights_init(cities)
    minisom.train_random(cities,1000)
    network = minisom.get_weights()
    np.savez_compressed('kaggle_tsp_minisom.npz', network = network)
    # print( network.shape)
    # route = get_route(cities,network)
    # print(route,len(route), len(set(route)))
    # print(route, len(route), len(set(route)))

    # adjacency_matrix = []
    # final_route=[]
    # for count in range(number_of_cities):

    #     row = []

    #     for inner_count in range(number_of_cities):
    #         row.append(calculate_distance(np.reshape(original_city[count],2), np.reshape(original_city[inner_count],2)))
    #         final_route.append(original_city[inner_count])
    #     adjacency_matrix.append(row)

    # sum = 0

    # for i in range(len(route) -2) :
    #     sum = sum + adjacency_matrix[i][i+1]

    # print(sum, final_route)

execute()