import numpy as np 
from keras.models import Sequential, load_model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.utils import plot_model
from keras import losses
from keras import optimizers
from keras.callbacks import Callback
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

FULL_TRAINING_PATH = 'CNN//Convolutional_Neural_Networks//dataset//training_set'
FULL_TESTING_PATH = 'CNN/Convolutional_Neural_Networks/dataset/test_set'

SMALL_TRAINING_PATH = 'CNN/Convolutional_Neural_Networks/smaller_set/training_set'
SMALL_TESTING_PATH = 'CNN/Convolutional_Neural_Networks/smaller_set/test_set'


def prepare_classifier(image_shape):

    classifier = Sequential()

    ## filters : the number of filters you want to specify.
    ## input_shape : only required in Conv2D is first layer. the size of images. 
    ## if WB image then (x,y,1), if RGB image then (x,y,3)

    classifier.add(Convolution2D(filters = 32, kernel_size = (3,3), \
                        input_shape = image_shape, activation = 'relu'))

    ## we add our subsampling layer
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    ## we will add another layer to COnvLayer to increase our accuracy numbers

    ## we no longer need to add input shape size
    ## for single value of kernel size, the same value will be used in both dimension
    classifier.add(Convolution2D(filters = 32, kernel_size = 3, \
                         activation = 'relu'))

    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    ## now that we have added our convolution layers and our pooling layers
    ## final step would be flatten the structure so that it can be fed into 
    ## artificial neural network

    ## how to flatten the structure: use Flatten

    ## after we flatten the structure the output shape of the model can be calculated.
    ## i.e the number of input to the first hidden layer in the ANN
    ## lets say we take a 32*32 image dimension , apply 64 filters to it
    ## then our output shape is 64*32*32 neurons
    classifier.add(Flatten())

    classifier.add(Dense(units = 64, activation = 'relu' ))

    classifier.add(Dense(units = 64, activation = 'relu' ))

    ## final layer. we are expecting binary classification hence units = 1
    ## and loss function as logistic sigmoid
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


    return classifier

def getTrainingData(path):
    """
        what this method would do is apply a bunch of changes to images
        and keep providing images in batches with combinations of changes 
        applied. changes could be rescaling, resizing, inverting, etc
    """
    training_data_generator = ImageDataGenerator(
                                                featurewise_center=True,
                                                featurewise_std_normalization=True,
                                                rotation_range=20,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                horizontal_flip=True,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                            )


    ## For flow_from_directory to work:  It should contain one subdirectory per class.
    ## this returns A keras_preprocessing.image.DirectoryIterator yielding tuples of `(x, y)`
    ## where `x` is a numpy array containing a batch
    ## of images with shape `(batch_size, *target_size, channels)`
    ## and `y` is a numpy array of corresponding labels

    training_set = training_data_generator.flow_from_directory(
                        path,
                        target_size=(64, 64),
                        batch_size=20,
                        class_mode='binary'
        )

    return training_set

def getTestingData(path):
    """
        we keep this image generator absolutely simple since 
        we would not want to touch the testing images but
        infact classify them as is.
    """
    testing_data_generator = ImageDataGenerator()

    test_set = testing_data_generator.flow_from_directory(
                        path,
                        target_size=(64, 64),
                        batch_size=20,
                        class_mode='binary'
        )

    print('Testing data found is :' , test_set.class_indices)
    print('Predictions to be made by model are : ', test_set.class_indices.values())

    return test_set

def fit( training_data, testing_data):

    classifier = KerasClassifier(build_fn = prepare_classifier)
    pipeline = Pipeline([
                ('keras_classifier', classifier)
        ])

    param_grid = {

        'keras_classifier__batch_size' : [10,20,30,50],
        'keras_classifier__epochs' : [100, 200, 300, 500],
        'keras_classifier__image_shape' : [training_data.image_shape],
        
    }


    grid = GridSearchCV(estimator = pipeline, param_grid = param_grid, n_jobs = -1)
    grid.fit_generator(
                            training_data,
                            steps_per_epoch = training_data.samples,
                            validation_data = testing_data,
                            validation_steps = testing_data.samples
                         )
    print("Best parameters are : ", grid.best_params_, '\n grid best score :', grid.best_score_)

    return grid.best_estimator_.model

def execute():
    """
        1. we fetch our images 
        2. segregate into testing and training set
        3. prepare the classifier
        4. compile the classifier
        5. pass the classifier through fit_generator
        6. check the results
    """
    
    
    training_data = getTrainingData(path = SMALL_TRAINING_PATH)
    testing_data = getTestingData(path = SMALL_TESTING_PATH)
    
    best_model = fit(training_data, testing_data)    

    metric_names = best_model.metric_names
    metric_values = best_model.metric_values

    for i, j in zip(metric_names, metric_values):

        print(i,':', j)
 
execute()