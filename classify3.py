"""
Created on Sun Jan 20 22:52:48 2019
@author: barbaraxiong
"""
import numpy as np
import tensorflow as tf
import time
import build_augmented_data
import build_augmented_data2
from PIL import Image, ImageOps
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from keras.applications import MobileNet, VGG16
from keras.applications.mobilenet import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras import Model
from keras.applications.resnet50 import ResNet50
from sklearn.metrics import confusion_matrix
from numpy import argmax
import cv2
seed = 7
matplotlib.use('pdf')

#_________________________________________________________________________________________________
# Running this everytime will augment data and randomly shuffle it

#data_sets = build_augmented_data.load_training_and_test_data()
#X_train, X_test, y_train, y_test = data_sets[0], data_sets[1], data_sets[2], data_sets[3]

data_sets = build_augmented_data2.load_training_and_test_data()
X_train, X_test, y_train, y_test = data_sets[0], data_sets[1], data_sets[2], data_sets[3]
og_split1,og_split2, og_split3, og_split4, og_split5 = data_sets[4], data_sets[5], data_sets[6], data_sets[7], data_sets[8]

print("DATA CHECKPOINT")

#_________________________________________________________________________________________________
def basic():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 160, 3)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def kaggle():
    '''from https://www.kaggle.com/hrmello/histology-with-cnn'''
    model = Sequential()
    model.add(Convolution2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(150, 150, 3)))
    model.add(Convolution2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Convolution2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=3))

    model.add(Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=3))

    model.add(Convolution2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=3))

    model.add(Convolution2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Convolution2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=3))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(160, 160, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def mobilenet():
    base_model = MobileNet(input_shape = (160,160,3), weights='imagenet',include_top=False)

    # imports the mobilenet model and discards the last 1000 neuron layer.

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    preds = Dense(8, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)
    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def resnet():

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    x = Dense(1024, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    predictions = Dense(8, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    adam = Adam(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

# model = VGG_16('vgg16_weights.h5')

#kfold = KFold(n_splits=5, shuffle=True, random_state=seed)



#results = cross_val_score(estimator, X_train, , cv=kfold, fit_params={"callbacks": [earlystop]})
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

#X_train, X_test, y_train, y_test = train_test_split(x, dummy_y, test_size=0.20)


estimator = KerasClassifier(build_fn=resnet, epochs=50, batch_size=8, verbose=1)
earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5,verbose=1, mode='auto')
estimator.fit(X_train, y_train, callbacks = [earlystop])

def work(X_train,X_test,y_train,y_test):

    encoder = LabelEncoder()
    encoder.fit(y_train)

    y_pred = estimator.predict(X_test)
    y_pred = encoder.inverse_transform(y_pred)

    y_test_label = []
    for i in range(len(y_test)):
        ya = argmax(y_test[i])
        y_test_label.append(ya)
    y_test = encoder.inverse_transform(y_test_label)
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    print("THE TEST ACCURACY: " + accuracy_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)

acc1 = work(X_train,X_test,y_train,y_test)

# work with og_split 1-5 and perform same operations to generate each of their X_train,.....y_test
# create acc2 - acc5 by passing it into work
# average the acc's and display final validation accuracy




classes = ['01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO', '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE', '08_EMPTY']

df_cm = pd.DataFrame(matrix, index = [i for i in classes],
                      columns = [i for i in classes])
plt.figure(figsize = (10,10))
svm = sn.heatmap(df_cm, annot=True)
figure = svm.get_figure()
figure.savefig('conf_resnet50_200_NEW.png', dpi=300)
