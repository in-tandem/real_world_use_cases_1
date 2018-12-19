import numpy as np 
from matplotlib import pyplot as plot 
from sklearn.preprocessing import MinMaxScaler
import random
from matplotlib import patches as patches

def create_input_data(number_of_rows, number_of_columns = 3):
    """

        Method returns a 3D data equal to number of rows.
        in this method we wil return a range of colors
        so each row will have random values for 0-255
        for red green blue channel

    """


    return np.random.randint(0,255, (number_of_rows, number_of_columns))


## our goal will be to visualize the above 3channel 3d color as a 
## 2D color map, with similar colors grouping together closely.
## so reds closes to reds, greens closer to green
## we will use a self organizing map to act as a dimensionality reduction
## technique. once reduced the network would be plotted on matplotlib
## to check if we have actually reduced the dimensions

def self_organize_map(x, size_of_grid, epoch, learning_rate, tau, sigma ):
    """
        runs a SOM on the given input data.

        :param x: the scaled input to the network

        :param size_of_grid: the size of the weight matrix. expecting a tuple (k,p)
                             remember the weight matrix is k*p*m dimension
                             where each k*p cell is associated with a vector
                             of length m

        :param epoch: int value depicting total number of iterations to be executed

        :param learning_rate: learning rate. for now we will use the same learning rate
                              across all iterations. however generally a decaying learning 
                              rate performs better

        :param tau: tau works as a decaying parameter. we will use the same tau parameter to
                    decay the neighbor hood radius as well as learning rate if applicable

        :param sigma: the sigma value will be used to depict the starting radius
                      after every iteration we will decay the radius using the tau parameter


    """
    
    dimensionality_of_input = x.shape[1]
    
    ## random weight matrix initialized.
    weight_matrix = np.random.rand(size_of_grid[0],size_of_grid[1],dimensionality_of_input)

    for _iteration in range(epoch):

        ## select a random x input
        x_random = x[random.randint(0, x.shape[0]-1)]

        ## compare values of weight matrix with the above selected x_random
        ## we will select the closest neuron - best matching unit BMU
        _best_matching_index = compare_node_with_weight_matrix(x_random,weight_matrix)

        ## once our BMU is selected. we will update the weights around it
        ## based on the neighborhood function as sigma parameter(As radius)
        ## and tau parameter( as decaying option)

        update_weights(x_random,_best_matching_index, weight_matrix,learning_rate,tau, sigma, _iteration)

    ## our weight_matrix is the network which has learnt the behavior and
    ## correlations between the values. we will plot this grid on a matplot
    ## for visualization aid

    return weight_matrix    


def calculate_distance(a, b):
    """
    this method would calculate the euclidean distance between
    points a and b.

    essentially this will serve as a distance measure between our
    node and weight neuron.

    """
    return np.linalg.norm(a - b)


def scale_input_data(x):
    """
    
    normalized input to the map to a range of 0-1.
    SOMs work best under scaled inputs.
    Using scikit learn MinMaXScaler. else we could have
    done manually as well.

    """

    return MinMaxScaler(feature_range = (0,1)).fit_transform(x)

def compare_node_with_weight_matrix(x, w):

    _units = {}
    
    for i, j in enumerate(w):
        count = 0
        
        for neuron in j:

            _units[(i,count)] = calculate_distance( x, neuron )
            count += 1

    ## sorting the dictionary according to value of each tuple
    ## where each tuple represents the co ordinates
    return sorted(_units, key = lambda x: _units[x])[0]

def update_weights(x, _best_matching_index, weight_matrix, learning_rate, tau, sigma, iteration):
    """

        how are the weights updated.
        based on sigma parameter, we select the sphere of influence
        w = w + learning_rate*neighborhood_function(sigma)*(x_random-w)

    """

    _best_matching_unit = weight_matrix[_best_matching_index[0]][_best_matching_index[1]]
    radius = decay_radius(sigma, tau, iteration)

    for i, j in enumerate(weight_matrix):
        
        for neuron in j:

            distance = calculate_distance( _best_matching_unit, neuron )

            if distance <= radius**2: ## within sphere of influence
                
                degree_of_influence = calculate_degree_of_influence(radius, distance)

                new_weight = neuron + (learning_rate*degree_of_influence*(x - neuron))

                neuron = new_weight

def decay_radius(radius, time_decay, iteration):
    """
        time decay operation on the initial radius value given

    """
    return radius * np.exp(- iteration/time_decay )

def calculate_degree_of_influence(radius, distance):
    """

    :param radius: the decayed value of radius
    :param distance: the distance between current weight cell and the best matching unit

    """
    return np.exp(- distance/ (2 * (radius)**2))

def plot_in_self_organized_map(net):
    fig = plot.figure()
    # setup axes
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, net.shape[0]+1))
    ax.set_ylim((0, net.shape[1]+1))
    ax.set_title('Self-Organising Map after iterations')

    # plot the rectangles
    for x in range(1, net.shape[0] + 1):
        for y in range(1, net.shape[1] + 1):
            ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                        facecolor=net[x-1,y-1,:],
                        edgecolor='none'))
    plot.show()

def execute():

    x = scale_input_data(create_input_data(100)) ## we create a 100 row, 3d color map and scale it

    epoch = 10000
    weight_shape = (4,4)
    learning_rate = 0.5
    tau = 0.2
    sigma = 12

    weight_matrix = self_organize_map(x, weight_shape,epoch,learning_rate,tau,sigma)

    plot_in_self_organized_map(weight_matrix)

execute()