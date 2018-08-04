import glob
import h5py
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, BatchNormalization
from keras.models import Model

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K

With_Data_Augmentation = False
N_CLASSES = 43
IMG_SIZE = 48
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def cnn_model(input_shape=None,NUM_CLASSES = None):

    inputs = Input(shape=input_shape)

    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    fc1 = Flatten()(pool3)
    fc1 = Dense(512, activation='relu')(fc1)
    fc1 = Dropout(0.5)(fc1)
    out = Dense(NUM_CLASSES, activation='softmax', name='fc1000')(fc1)

    model = Model(input=inputs, output=out)

    return model


with h5py.File('./data/GTSRB_train.h5') as hf:
    X, Y = hf['imgs'][:], hf['labels'][:]
print("Loaded images from GTSRB_train.h5")

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

batch_size = 32
nb_epoch = 30


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))


model = cnn_model(input_shape=INPUT_SHAPE, NUM_CLASSES=N_CLASSES)
# let's train the model using SGD + momentum (how original).
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

json_string = model.to_json()
open('./models/GTSRB_classifier_architecture.json', 'w').write(json_string)
checkpointer = ModelCheckpoint(filepath='./models/GTSRB_classifier_best_weights.h5', verbose=1, monitor='loss',
                               mode='auto', save_best_only=True)

if With_Data_Augmentation is True:

    datagen = ImageDataGenerator(featurewise_center=False,
                                featurewise_std_normalization=False,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=0.2,
                                shear_range=0.1,
                                rotation_range=10)

    datagen.fit(X_train)

    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        epochs=nb_epoch,
                        validation_data=(X_val, Y_val),
                        verbose=1,
                        callbacks=[LearningRateScheduler(lr_schedule), checkpointer])

else:

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=(X_val, Y_val),
              shuffle=True,
              verbose=1,
              callbacks=[LearningRateScheduler(lr_schedule), checkpointer])


model.save_weights('./models/GTSRB_classifier_last_weights.h5', overwrite=True)
