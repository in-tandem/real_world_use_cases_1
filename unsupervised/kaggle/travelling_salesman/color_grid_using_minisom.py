import numpy as np 
import pandas as panda 
from minisom import MiniSom
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
def scale_input_data(x):
    """
    
    normalized input to the map to a range of 0-1.
    SOMs work best under scaled inputs.
    Using scikit learn MinMaXScaler. else we could have
    done manually as well.

    """
    return MinMaxScaler(feature_range = (0,1)).fit_transform(x)

input_data = scale_input_data(create_input_data(198000))
minisom = MiniSom(15,15, 3,sigma=1.0,learning_rate=0.5)
minisom.random_weights_init(input_data)
minisom.train_random(input_data,10000)
plot_in_self_organized_map(minisom.get_weights())