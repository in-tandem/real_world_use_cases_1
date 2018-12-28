import pandas as panda 
import numpy as np 
from scipy import spatial
from copy import deepcopy
import itertools, time, datetime


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


FILE_NAME = 'all/cities.csv'


data = panda.read_csv(FILE_NAME)


cities = []

for i in range(len(data)):
    cities.append([data.iloc[i]['X'], data.iloc[i]['Y']])


number_of_cities = len(cities)

original_city = deepcopy(cities)

start_node  = cities[0]
navigation_map = [start_node]
route = [0]

def find_nearest(point, network):

    nearest, index = spatial.KDTree(network).query(point)
    return network[index]

for i in range(number_of_cities -1):

    training_timer       = CodeTimer('training')
    
    with training_timer:

        nearest = find_nearest(start_node, [i for i in cities if i not in navigation_map])
        # nearest = find_nearest(start_node, list(filter(lambda x:x not in navigation_map, cities)))
        navigation_map.append(nearest)
        # c = np.where((data.X==nearest[0]) & (data.Y==nearest[1]))[0][0]
        # route.append(c)
        # print('found nearest point ', nearest, c)
        start_node = nearest

    print(training_timer.took)

print(navigation_map, len(np.unique(navigation_map,axis=0)))
np.savez_compressed('kdtree.npz', route = navigation_map)