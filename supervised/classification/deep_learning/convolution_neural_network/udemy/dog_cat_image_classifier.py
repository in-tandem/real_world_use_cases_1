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
from skimage import io


FULL_TRAINING_PATH = 'CNN//Convolutional_Neural_Networks//dataset//training_set'
FULL_TESTING_PATH = 'CNN/Convolutional_Neural_Networks/dataset/test_set'

SMALL_TRAINING_PATH = 'CNN//Convolutional_Neural_Networks//smaller_set//training_set'
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

    print('Testing data found is :' , test_set.class_indices)
    print('Predictions to be made by model are : ', test_set.class_indices.values())

    return test_set



## lets create a Callback object to check the predictions made by keras
## at the end of each epoch

class PredictionLogger(Callback):

    def on_epoch_end(self, epoch, logs={}):
        """
            len(self.validation_data) == 3, because 
            validation_data[0] == train_x (that you input in model.fit()), 
            validation_data[1] == train_y, 
            validation_data[2]=sample_weight,

        validation_data needs to be provided in the fit method
        else self.validation_data will be empty

        """
        val_predict = self.model.predict(self.validation_data[0])
        # val_predict = (val_predict > 0.5) ##using the same activation function
        val_targ = self.validation_data[1]
        print("val_predict: ", val_predict)
        print("actual :", val_targ)
        return


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
    
    classifier = prepare_classifier(training_data.image_shape)

    ##another way
    # classifier.compile(optimizer = 'adam', loss = losses.binary_crossentropy, metrics = ['accuracy'])


    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    ## steps_per_epoch = number of data provided. in our case it is 8k images in training set
    classifier.fit_generator(
                            training_data,
                            steps_per_epoch = training_data.samples,
                            epochs = 10,
                            validation_data = testing_data,
                            validation_steps = testing_data.samples
                            # callbacks = [PredictionLogger()]
                         )

    classifier.save('cats_or_dogs.h5py')
    ## commented for now since for now we are only trying out for 1 epoch
    # plotResultsAcrossEpoch(classifier.history.history.get(''), 1, \
    #                   xlabel = '', ylabel = '', title = '')
    
    ## lets predict one image based on the classifier created above
    ## we will compare the validation accuracies

    image = load_img(
        'CNN/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_1.jpg',
        target_size = training_data.image_shape
    )

    print(type(image))

    ## all neural networks execute in batch. and hence consider inputs in batches
    ## the dim of the input to the neural network, hence needs to be (batch , abc)
    ## where abc is the dimension of the input eg 2d or 3d
    
    # image = np.expand_dims(img_to_array(image), axis = 0)
    # y_true = 1 # image is of a dog

    # y_predict = classifier.predict(image)

    # print(y_predict)


execute()