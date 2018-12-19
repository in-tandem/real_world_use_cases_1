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

FILE_NAME = 'all/cities.csv'

class Location(object):

    def __init__(self, identifier, x, y):
        self.identifier = identifier
        self.x = x
        self.y = y

def calculate_distance(a, b):
    
    return round(sqrt(  (a.x - b.x)**2 + (a.y - b.y)**2  ))

data = panda.read_csv(FILE_NAME)

header = data.columns.values

locations = []
# locations.append(Location(int(header[0]), float(header[1]), float(header[2])))
# data.columns = ['id','x','y']

for i in range(len(data)):

    locations.append(Location(data.iloc[i]['CityId'],data.iloc[i]['X'],data.iloc[i]['Y']))

print(len(locations))

number_of_cities = len(locations)

adjacency_matrix= []

for count in range(number_of_cities):

    row = []

    for inner_count in range(number_of_cities):
        row.append(calculate_distance(locations[count], locations[inner_count]))

    adjacency_matrix.append(row)

print(adjacency_matrix)

cities = list(range(0,number_of_cities))
print(cities)



def execute_djibuouti(adjacency_matrix, number_of_djibouti_scities, sample):

    starting_point = 0

    epoch = 1000

    record = []

    for i in range(epoch):
       
       ## in cases where starting_point is not given it is important to check if 
       ## algorithm can create the optimum route
        population = Population(number = 197768, \
                                dna_size = number_of_djibouti_scities, \
                                sample = sample, \
                                mutation_rate = 0.1, \
                                starting_point = starting_point, \
                                adjacency_matrix = adjacency_matrix)

        population.populate()
        
        print('starting genetic mutations....')
        count = 1

        while True:
            
            ## recalculation fitness scores happens over new mating pool
            flag = population.calculate_fitness()
            
            if flag:
                print('Found phrase. solution found in generation: ', count)
                cost, path = population.record.get_lowest_score_and_path()
                print('Path: ', path , ' cost: ', cost)
                record.append((cost, path))
                break
        
            else:
                
                print('Existing mutation failed. Starting natural selection and cross over' )
                population.natural_selection()
        
                print('Natural selection and cross over completed. proceeding to check again')
        
            count = count + 1

    print('Finding lowest score...')
    cost = min(record, key = lambda x:x[0])

    print('minimum cost is ', cost[0], ' path is ', list(filter(lambda x: x == cost, record))[0][1])

def cost_function(adjacency_matrix,starting_point, points, end_point, sum_distance = 0):
    """

    :param points : sequence of other locations to be scanned, if length is one. means we can calc the distaince
    :param end_point : current end point. ie dist(starting_point, end_point). end_point is returned so we can show the path
    :param sum_distance: sum of cost of distance travelled so far
    """
 
    end_point = [] if end_point is None else end_point

    if len(points) == 1:
        sum_distance = sum_distance + adjacency_matrix[starting_point-1][points[0]-1]
        end_point.append(points[0])
        return sum_distance  ,end_point

    else:

        ## you select a pool with all possibilities except a single element k and statting point s
        ## calculate cost of each
        ## select the ones with lowest cost
        ## in case , pool selections are greater than single value, recursion would apply
        print('finding minimum again', sum_distance)
        return min(map(lambda x :cost_function(adjacency_matrix,starting_point,[x], end_point + [x],sum_distance) + \
                cost_function(adjacency_matrix, x, [i for i in points if i !=x and i!= starting_point],end_point + [x],sum_distance),\
                 points))


def execute_djibouti_using_dynamic_programming(adjacency_matrix, number_of_cities, cities):
    
    starting_point = 1
    cost_and_path = cost_function(adjacency_matrix,starting_point, cities, None, sum_distance = 0)

    print('shortest cost : ', sum(cost_and_path[::2])+ \
    adjacency_matrix[starting_point-1][cost_and_path[-1][-1]-1], \
        ' \n path: ', cost_and_path[-1])

execute_djibuouti(adjacency_matrix, number_of_cities, cities)


# execute_djibouti_using_dynamic_programming(adjacency_matrix, number_of_cities, cities)