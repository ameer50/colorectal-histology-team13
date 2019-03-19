#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:24:35 2019

@author: barbaraxiong
"""

import imageio
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from random import shuffle

def load_batch(foldername):
    '''load data from single folder'''
    images = []
    labels = []
    for category in os.listdir(foldername):
        if os.path.isdir(os.path.join(foldername, category)):
            for img in os.listdir(os.path.join(foldername, category)):
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    image = Image.open(os.path.join(foldername, category)+"/"+img)
                    sess = tf.Session()
                    image = image.resize((128,128))
                    with sess.as_default():
                        images.append(np.asarray(image))
                    labels.append(str(category))
    print(np.asarray(images))
    return np.asarray(images), np.asarray(labels)

def load_data():
    '''load all folder data and merge training batches'''
    X, Y = load_batch("colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/")
    # For only inputting part of the dataset for test runs
#    shuffle(X)
#    shuffle(Y)
#    z = list(zip(X,Y))
#    shuffle(z)
#    X, Y = zip(*z)
    xs = X#[:5000]
    ys = Y#[:5000]
    xs = np.concatenate([xs])
    ys = np.concatenate([ys])
    classes = ['01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO', '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE', '08_EMPTY']

    # Normalize Data
    def norm_image(image):
        return (image - np.mean(image)) / np.std(image)
    for i in range(len(xs)):
        xs[i] = norm_image(xs[i])
    #mean_image = np.mean(xs, axis=0)
    #xs = np.subtract(xs, mean_image, out=xs, casting = "unsafe")

    data_dict = {
        'images': xs,
        'labels': ys
        }
    return data_dict

def main():
    data_sets = load_data()

    
if __name__ == '__main__':
  main()

