#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:59:52 2018

@author: shidiyang
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

dir = os.path.dirname(__file__) #need run the file first
img_width = 150
img_height = 150
"""
:param
p: Dropout rate (disable how many neurons in this layer)
"""
def create_model(p, input_shape=(32, 32, 3)):
    # Part1 - Initializing the CNN
    classifier = Sequential()
    #step1 - Convolution
    classifier.add(Convolution2D(32, (3, 3), padding = 'same', input_shape = input_shape, activation = 'relu'))
    #step2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    #Increase testset accuracy and avoid overfitting by adding a second conv layer
    """
    Convolution2D(nb_filter, nb_rows, nb_cols)
    nb_filter: the number of feature detectors
    nb_rows, nb_cols: the number of rows and cols of a feature map
    common practice:
    start with 32 feature detectors,
    add more layers 64..
    add 128..
    add 256..
    """
    classifier.add(Convolution2D(32, (3, 3), padding = 'same', activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(64, (3, 3), padding = 'same', activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(64, (3, 3), padding = 'same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    #step3 - Flattening
    classifier.add(Flatten())

    #step4 - Full Connection
    #Dense(): fully connected
    """
    output_dim: the number of nodes in the hidden layer
    common practice: 
    choose the number between the number of input nodes and the number of output nodes
    """
    classifier.add(Dense(output_dim = 64, activation = 'relu'))
    classifier.add(Dropout(p)) #each neuron in this layer has p% probability be dropoutted
    classifier.add(Dense(output_dim=64, activation='relu'))
    classifier.add(Dense(output_dim=64, activation='relu'))
    classifier.add(Dropout(p/2))
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

    #optimizer = 'adam'
    optimizer = Adam(lr=1e-3)
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#Part2 - Fitting the CNN to the images
def run_training(bs=32, epochs=10):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(
        os.path.join(dir, '.\\dataset\\training_set'),
        target_size=(img_height, img_width),
        batch_size=bs,
        class_mode='binary')

    test_set = test_datagen.flow_from_directory(
        os.path.join(dir, '.\\dataset\\test_set'),
        target_size=(img_height, img_width),
        batch_size=bs,
        class_mode='binary')

    classifier = create_model(p=0.6, input_shape=(img_height, img_width, 3))

    classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/bs,
        epochs=epochs,
        validation_data=test_set,
        nb_val_samples=2000/bs)

def test_single_image(classifier):
    import sys
    from PIL import Image
    sys.modules['Image'] = Image

    # save model
    classifier.save('classifier_model.h5', overwrite=True)
    del classifier

    from keras.models import load_model
    classifier = load_model(os.path.join(dir, '.\\classifier_model.h5'))

    # Single Prediction
    test_image = image.load_img(os.path.join(dir, '.\\dataset\\single_prediction\\2.jpg'),
                                target_size=(64, 64))

    # required input shape: 64*64*3
    test_image = image.img_to_array(test_image)

    # add one dims, cause predict() only accept batch variables (1, 64, 64, 3)
    test_image = np.expand_dims(test_image, axis=0)

    result = classifier.predict(test_image)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'


def main():
    run_training(bs=128, epochs=100)

if __name__=="__main__":
    main()
