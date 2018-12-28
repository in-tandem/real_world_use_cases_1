import numpy as  np 
from self_organizing_map import self_organize_map 
NORMALIZED_X_Y_PATH = 'normalized_kaggle_x_y.npz'

## the pickled form of cities has the entire data set including starting and ending position
## we know that our starting and ending node is the first city.
## so what we do is we will extract out the first node. and find the best least costing route
## among the remaining cities. once we find that we will just add the cost distance between
## first city and the network plot created

## we will attempt to use our existing SOM created in the model self_organizing_map.py

def execute():
    """

        1. reading the pickled form of the data. and keeping aside the first element.
           our data contains 2 points . so essentially 198k * 2. so our m is 2.

        2. we will create a network of say 198k * 2. each has 2 points bcoz m = 2.


    IMP TO NOTE IN POINT 3: SOM always achieves a dimensionality reduction. If you notice our grid
    size here is (198k*4, 1). Our original data set had (198k,2) dimension. where each of the 198k 
    had 2 dimensions. however using our SOM technique we reduce it down to 1d, is each of the 198k
    will be associated with 1 point. In our case that particular point will be the point closest to
    the city in input. 

    How do we derive the route once the network learns??    



    """
    # 1. reading the saved pickled data and settin aside the first element

    cities = np.load(NORMALIZED_X_Y_PATH)
    cities = cities.get('cities')
    starting_point = cities[0]

    number_of_cities = len(cities) - 1

    # 2. create a network and feed to SOM

    epoch = 5000
    weight_shape = (number_of_cities, 1)
    learning_rate = 1.2
    tau = 1000
    sigma = 2000 ## initial radius will be half way of the weight matrix dimension (5x,5y) /2 = 2.5

    print('starting to SOM....')
    weight_matrix = self_organize_map(cities[1:], weight_shape,epoch,learning_rate,tau,sigma, iteration_max= 800)

    print('Starting to pickle output....')
    np.savez_compressed('normalized_network_1time.npz', network = weight_matrix)


execute()