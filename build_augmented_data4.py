#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 1 2019
@author: ameer syedibrahim
"""
import imageio
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from random import shuffle
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os, shutil
import random
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import cv2
import imgaug
from imgaug import augmenters

def load_training_and_test_data():
    '''load all folder data and merge training batches'''

    new_dirr = "colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/"

    def gen_labels(label, a_list):

        labels = []
        for x in range(len(a_list)):
            labels.append(label)

        return labels

    # ______________________________________________________________________________________

    def str_to_img(the_dir, a_list):  # returns list of all images

        all_images = []
        for el in a_list:
            img = image.load_img(the_dir + "/" + el).resize((150, 150))  # this is a PIL image
            x = image.img_to_array(img)  # this is a Numpy array with shape (150, 150, 3)
            all_images.append(x)

        return all_images

    # ______________________________________________________________________________________

    # normalize each image
    def norm_image(image):
        return (image - np.mean(image)) / np.std(image)

    # randomly augment an image into 3 new images
    def augment(X):
        seq = augmenters.Sequential(
            [
                augmenters.Fliplr(0.5),  # flips 50% of all images horizontally
                augmenters.Flipud(0.5),  # flips 50% of all images vertically
                augmenters.Affine(rotate=(-20, 20)),  # rotate images by -20 to 20 degrees
                augmenters.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                # images are translated by -20% to 20% on both x-axis and y-axis independently
                augmenters.Affine(scale=(0.9, 1.1)),  # scale images from 90% to 110% scale
            ]
        )
        return seq.augment_images([X, X, X, X, X])

    print("CHECKPOINT1: Iterating over every label...")
    the_images = []
    the_labels = []
    vval = os.listdir(new_dirr)
    for the_fi in vval:  # iterate over each label 01_TUMOR .... 08_EMPTY
        star_dir = new_dirr + the_fi  # make note of the directory that now includes this folder
        the_list = []
        for filename in os.listdir(star_dir):
            the_list.append(filename)  # make the_list contain string names of all the images in the label folder

        the_images = the_images  + str_to_img(star_dir, the_list)
        the_labels = the_labels + gen_labels(the_fi, the_list)
        
    temp_list = list(zip(the_images,the_labels))
    random.shuffle(temp_list)
    r_list = list(zip(*temp_list))
    the_images = list(r_list[0])
    the_labels = list(r_list[1])

    for i in range(len(the_images)):
        the_images[i] = norm_image(the_images[i])

    print("CHECKPOINT2: Creating batches for 5-fold validation...")

    print(len(the_images))
    print(the_images[0])
    print(len(the_labels))
    print(the_labels[0])
    batch1i = the_images[:1000]  # split into 5 categories
    batch2i = the_images[1000:2000]  # split into 5 categories
    batch3i = the_images[2000:3000]  # split into 5 categories
    batch4i = the_images[3000:4000]  # split into 5 categories
    batch5i = the_images[4000:]  # split into 5 categories

    batch1l = the_labels[:1000]  # split into 5 categories
    batch2l = the_labels[1000:2000]  # split into 5 categories
    batch3l = the_labels[2000:3000]  # split into 5 categories
    batch4l = the_labels[3000:4000]  # split into 5 categories
    batch5l = the_labels[4000:]  # split into 5 categories


    print("CHECKPOINT3: Performing augmentation on Batches 1-5...")

    the_test_images1 = batch1i
    the_test_labels1 = batch1l
    the_train1 = batch2i + batch3i + batch4i + batch5i
    the_labels1 = batch2l + batch3l + batch4l + batch5l
    the_augmented_train1 = []
    the_augmented_labels1 = []

    for el in the_train1:
        the_augmented_train1 = the_augmented_train1 + augment(el)

    for cz in the_labels1:
        the_augmented_labels1 = the_augmented_labels1 + [cz]*5


    # (VERIFY SHAPE)
    # the_augmented_train1 contains (4000*3 ?) images that are now augmented
    ## the_augmented_labels1 contains (4000*3) labels for the above images
    # the_test_images1 contains (1000) images
    # the_test_labels1 contains (1000) labels

    '''
    the_test_images2 = batch2i
    the_test_labels2 = batch2l
    the_train2 = batch1i + batch3i + batch4i + batch5i
    the_labels2 = batch1l + batch3l + batch4l + batch5l
    the_augmented_train2 = []
    the_augmented_labels2 = []

    for el in the_train2:
        the_augmented_train2 = the_augmented_train2 + augment(el)

    for cz in the_labels2:
        the_augmented_labels2 = the_augmented_labels2 + [cz]*5

    the_test_images3 = batch3i
    the_test_labels3 = batch3l
    the_train3 = batch1i + batch2i + batch4i + batch5i
    the_labels3 = batch1l + batch2l + batch4l + batch5l
    the_augmented_train3 = []
    the_augmented_labels3 = []

    for el in the_train3:
        the_augmented_train3 = the_augmented_train3 + augment(el)

    for cz in the_labels3:
        the_augmented_labels3 = the_augmented_labels3 + [cz]*5


    the_test_images4 = batch4i
    the_test_labels4 = batch4l
    the_train4 = batch1i + batch2i + batch3i + batch5i
    the_labels4 = batch1l + batch2l + batch3l + batch5l
    the_augmented_train4 = []
    the_augmented_labels4 = []

    for el in the_train4:
        the_augmented_train4 = the_augmented_train4 + augment(el)

    for cz in the_labels4:
        the_augmented_labels4 = the_augmented_labels4 + [cz]*5



    the_test_images5 = batch5i
    the_test_labels5 = batch5l
    the_train5 = batch1i + batch2i + batch3i + batch4i
    the_labels5 = batch1l + batch2l + batch3l + batch4l
    the_augmented_train5 = []
    the_augmented_labels5 = []

    for el in the_train5:
        the_augmented_train5 = the_augmented_train5 + augment(el)

    for cz in the_labels5:
        the_augmented_labels5 = the_augmented_labels5 + [cz]*5


   #__________________________________________________________________
    '''
    print("CHECKPOINT4: Shuffle 5 Batches of Training Data...")
    # SHUFFLE TRAINING DATA
    temp1 = list(zip(the_augmented_train1, the_augmented_labels1))
    random.shuffle(temp1)
    f_list1 = list(zip(*temp1))
    the_augmented_train1 = np.asarray(list(f_list1[0]))
    the_augmented_labels1 = np.asarray(list(f_list1[1]))

    '''
    temp2 = list(zip(the_augmented_train2, the_augmented_labels2))
    random.shuffle(temp2)
    f_list2 = list(zip(*temp2))
    the_augmented_train2 = np.asarray(list(f_list2[0]))
    the_augmented_labels2 = np.asarray(list(f_list2[1]))

    temp3 = list(zip(the_augmented_train3, the_augmented_labels3))
    random.shuffle(temp3)
    f_list3 = list(zip(*temp3))
    the_augmented_train3 = np.asarray(list(f_list3[0]))
    the_augmented_labels3 = np.asarray(list(f_list3[1]))

    temp4 = list(zip(the_augmented_train4, the_augmented_labels4))
    random.shuffle(temp4)
    f_list4 = list(zip(*temp4))
    the_augmented_train4 = np.asarray(list(f_list4[0]))
    the_augmented_labels4 = np.asarray(list(f_list4[1]))

    temp5 = list(zip(the_augmented_train5, the_augmented_labels5))
    random.shuffle(temp5)
    f_list5 = list(zip(*temp5))
    the_augmented_train5 = np.asarray(list(f_list5[0]))
    the_augmented_labels5 = np.asarray(list(f_list5[1]))
    '''
   #__________________________________________________________________
    # SHUFFLE TEST DATA
    print("CHECKPOINT5: Shuffle 5 Batches of Test Data...")

    temp11 = list(zip(the_test_images1, the_test_labels1))
    random.shuffle(temp11)
    f_list11 = list(zip(*temp11))
    the_test_images1 = np.asarray(list(f_list11[0]))
    the_test_labels1 = np.asarray(list(f_list11[1]))
    '''
    temp22 = list(zip(the_test_images2, the_test_labels2))
    random.shuffle(temp22)
    f_list22 = list(zip(*temp22))
    the_test_images2 = np.asarray(list(f_list22[0]))
    the_test_labels2 = np.asarray(list(f_list22[1]))

    temp33 = list(zip(the_test_images3, the_test_labels3))
    random.shuffle(temp33)
    f_list33 = list(zip(*temp33))
    the_test_images3 = np.asarray(list(f_list33[0]))
    the_test_labels3 = np.asarray(list(f_list33[1]))

    temp44 = list(zip(the_test_images4, the_test_labels4))
    random.shuffle(temp44)
    f_list44 = list(zip(*temp44))
    the_test_images4 = np.asarray(list(f_list44[0]))
    the_test_labels4 = np.asarray(list(f_list44[1]))

    temp55 = list(zip(the_test_images5, the_test_labels5))
    random.shuffle(temp55)
    f_list55 = list(zip(*temp55))
    the_test_images5 = np.asarray(list(f_list55[0]))
    the_test_labels5 = np.asarray(list(f_list55[1]))
    '''

    # __________________________________________________________________

    #X_trains = [the_augmented_train1, the_augmented_train2, the_augmented_train3, the_augmented_train4, the_augmented_train5]
    #X_tests = [the_test_images1, the_test_images2, the_test_images3, the_test_images4, the_test_images5]
    #y_trains = [the_augmented_labels1, the_augmented_labels2, the_augmented_labels3, the_augmented_labels4, the_augmented_labels5 ]
    #y_tests = [the_test_labels1, the_test_labels2, the_test_labels3, the_test_labels4, the_test_labels5 ]



    #final_list = [X_trains, X_tests, y_trains, y_tests]
    print("CHECKPOINT6: Data Preparation Complete!")
    return [the_augmented_train1, the_augmented_labels1,the_test_images1,the_test_labels1]


def main():
    data_sets = load_training_and_test_data()


if __name__ == '__main__':
    main()

