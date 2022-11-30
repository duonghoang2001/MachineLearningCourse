# Project 5: Corgi Classification with Convolutional Neural Networks
# Author: Duong Hoang
# CS 460G - 001
# Due Date: May 3rd, 2022

'''
    Purpose: categorize Corgi
    Pre-cond: Corgi images
    Post-cond: CNN model, Corgi classification

'''

### Implementation ###

# initialize
TRAIN_DATA_PATH = "./data/training_data"
TEST_DATA_PATH = "./data/testing_data"
IMAGE_SIZE = 150
CLASSES = ["cardigan", "pembroke"]
VALIDATION_SIZE = 0.2
EPOCHS = 10
BATCH_SIZE = 30

# import libraries
from dataset import *
import tensorflow as tf
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator

data_sets = read_train_sets(TRAIN_DATA_PATH, IMAGE_SIZE, CLASSES, VALIDATION_SIZE)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
history = model.fit(data_sets.train.images, data_sets.train.labels, batch_size=BATCH_SIZE,
                    epochs=EPOCHS)

model.evaluate(data_sets.valid.images, data_sets.valid.labels)






