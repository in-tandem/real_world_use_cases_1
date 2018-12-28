import numpy as np 
from self_organizing_map import compare_node_with_weight_matrix,calculate_distance
import pandas as panda 
from multiprocessing import Pool
import random

FILE_NAME = 'all/cities.csv'

def get_original_cities():

    data = panda.read_csv(FILE_NAME)

    cities = []

    for i in range(len(data)):
        cities.append([data.iloc[i]['CityId'],data.iloc[i]['X'], data.iloc[i]['Y']])

    return cities

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

        index = compare_node_with_weight_matrix(city, network)

        route[count] = index 
        # print('i found index: ', index)
        count+=1

    # return route    
    return sorted(route, key = lambda x: route[x][0])


def get_route_parallelism(cities, network):
    """
    
    For each city, calculate the neuron closest resembling
    it. say it is n1. find the index of n1 in the weight matrix
    .
    the route will cities , sorted by the order of appearance
    of their matching neuron n1 in the weight matrix
    
    """
    
    route = []
    count = 0
    with Pool(8) as p:
        route = p.starmap(compare_node_with_weight_matrix, [(i,network) for i in cities], 10)

    # for city in cities:

    #     route.append(compare_node_with_weight_matrix(city, network))

    print('foudn route: ', route)    
    return sorted(route)



cities = np.load('normalized_kaggle_x_y.npz').get('cities')
network = np.load('normalized_network_1time.npz').get('network')

start_node = cities[0]

if __name__ == '__main__':

    print(len(cities))

    route = get_route_parallelism(cities[1:], network)

    original_cities = get_original_cities()

    # sum = np.linalg.norm(start_node - original_cities[1])

    my_final_route = [0]

    for i in range(len(route) -2) :

        start_city = original_cities[i][0]
        end_city = original_cities[i+1][0]
        my_final_route.append(start_city)
        my_final_route.append(end_city)

        print('start city index ', start_city, ' end city index :', end_city)

        # sum = sum + np.linalg.norm(start_city-end_city)

    my_final_route.append(0)

    # sum = sum + np.linalg.norm(original_cities[len(original_cities)-1]-start_city)

    np.savez_compressed('route.npz', route = my_final_route)

    print(my_final_route)

# print(sum)
