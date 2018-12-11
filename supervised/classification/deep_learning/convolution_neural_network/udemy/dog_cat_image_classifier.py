import numpy as np 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import losses
from keras import optimizers

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

    ## final layer. we are expecting binary classification hence units = 1
    ## and loss function as logistic sigmoid
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

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

    return test_set


def execute():
    """
        1. we fetch our images 
        2. segregate into testing and training set
        3. prepare the classifier
        4. compile the classifier
        5. pass the classifier through fit_generator
        6. check the results
    """
    
    
    training_data = getTrainingData(path = 'CNN//Convolutional_Neural_Networks//dataset//training_set')
    testing_data = getTestingData(path = 'CNN/Convolutional_Neural_Networks/dataset/test_set' )
    
    classifier = prepare_classifier(training_data.image_shape)

    ##another way
    # classifier.compile(optimizer = 'adam', loss = losses.binary_crossentropy, metrics = ['accuracy'])


    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    ## steps_per_epoch = number of data provided. in our case it is 8k images in training set
    classifier.fit_generator(
                            training_data,
                            steps_per_epoch = training_data.samples,
                            epochs = 1,
                            validation_data = testing_data,
                            validation_steps = testing_data.samples
                         )

    ## TODO
    ## check.. is the history still available
    ## check.. use one image to check for predict function
    ## check.. when you predict see what it returns.
    ## check.. if its a number we may need to specify a threshold value for which is cat/dog
    ## 
    # plot_model(classifier, to_file='image_classifier_model.png')
    ## check the return values from image data generators

execute()