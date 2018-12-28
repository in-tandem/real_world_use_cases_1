import pandas as panda 
import numpy as np 
from scipy import spatial
from copy import deepcopy

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

start_node  = cities[0]
navigation_map = [start_node]

def find_nearest(point, network):

    nearest, index = spatial.KDTree(network).query(point)
    return network[index]

for i in range(number_of_cities -1):

    nearest = find_nearest(start_node, [i for i in cities if i not in navigation_map])
    navigation_map.append(nearest)
    start_node = nearest

print(navigation_map, len(np.unique(navigation_map,axis=0)))