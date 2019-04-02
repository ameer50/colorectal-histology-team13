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


def load_batch(foldername):
    '''load data from single folder'''
    images = []
    labels = []
    for category in os.listdir(foldername):
        if os.path.isdir(os.path.join(foldername, category)):
            for img in os.listdir(os.path.join(foldername, category)):
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    image = Image.open(os.path.join(foldername, category) + "/" + img)
                    sess = tf.Session()
                    image = image.resize((150, 150))
                    with sess.as_default():
                        images.append(np.asarray(image))
                    labels.append(str(category))
    print(np.asarray(images))
    return np.asarray(images), np.asarray(labels)


def load_training_and_test_data():
    '''load all folder data and merge training batches'''

    new_dirr = "colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/"
    dict1 = {}
    dict2 = {}

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

    # in a folder 'foldername', each file is renamed to a name that starts with 'tumor_type' followed by a number

    def rename_for_folder(foldername, tumor_type):
        i = 0
        for filename in os.listdir(foldername):
            dst = tumor_type + str(i) + ".jpg"
            src = foldername + '/' + filename
            dst = foldername + '/' + dst
            i += 1
            # rename() function will
            # rename all the files
            os.rename(src, dst)

        # ______________________________________________________________________________________

    # normalize each image
    def norm_image(image):
        return (image - np.mean(image)) / np.std(image)

    # for every image x, augmentations are performed on it and the resulting images are stored in the savedir
    # OLD AUGMENTATION METHODS
    '''def augment7(the_dir, savedir):
        datagen3 = ImageDataGenerator( horizontal_flip=True, vertical_flip=True)
        img = image.load_img(the_dir).resize((150, 150))  # this is a PIL image
        x = image.img_to_array(img)  # this is a Numpy array with shape (150, 150, 3)
        x = norm_image(x);
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        i = 0
        for batch in datagen3.flow(x, batch_size=1, save_to_dir=savedir, save_prefix='new', save_format='jpg'):
            i += 1
            if i > 6:
                break

    def augment5(the_dir, savedir):
        datagen3 = ImageDataGenerator()
        unique = the_dir.split("/")[4]
        img = image.load_img(the_dir).resize((150, 150))  # this is a PIL image
        x = image.img_to_array(img)  # this is a Numpy array with shape (150, 150, 3)
        # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        x90 = datagen3.apply_transform(x, {'theta': 90})
        x180 = datagen3.apply_transform(x, {'theta': 180})
        x270 = datagen3.apply_transform(x, {'theta': 270})
        xhflip = datagen3.apply_transform(x, {'flip_horizontal': True})
        xvflip = datagen3.apply_transform(x, {'flip_vertical': True})
        cv2.imwrite(os.path.join(savedir, unique + '90.jpg'), x90)
        cv2.imwrite(os.path.join(savedir, unique + '180.jpg'), x180)
        cv2.imwrite(os.path.join(savedir, unique + '270.jpg'), x270)
        cv2.imwrite(os.path.join(savedir, unique + 'h.jpg'), xhflip)
        cv2.imwrite(os.path.join(savedir, unique + 'v.jpg'), xvflip) '''

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
        return seq.augment_images(X*5)

        # ______________________________________________________________________________________

        # ______________________________________________________________________________________
        # make all this garbo cleeeean
    grand_test_final = []
    grand_test_labels = []
    
    for the_fi in os.listdir(new_dirr):  # iterate over each label 01_TUMOR .... 08_EMPTY
        star_dir = new_dirr + the_fi  # make note of the directory that now includes this folder
        the_list = []
        the_images = []
        the_labels = []
        for filename in os.listdir(star_dir):
            the_list.append(filename)  # make the_list contain string names of all the images in the label folder
            
        the_images.append(str_to_img(star_dir, the_list))
        the_labels.append(gen_labels(the_fi, the_list))
        
    temp = list(zip(the_images,the_labels))
    random.shuffle(temp)
    r_list = list(zip(*temp_list))
    the_images = list(r_list[0])
    the_labels = list(r_list[1])

    for i in range(len(the_images)):
        the_images[i] = norm_image(the_images[i])
    
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
    # the_augmented_train1 contains (4000*5 ?) images that are now augmented
    ## the_augmented_labels1 contains (4000*5) labels for the above images
    # the_test_images1 contains (1000) images
    # the_test_labels1 contains (1000) labels

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


    # (VERIFY SHAPE)
    # the_augmented_train1 contains (4000*5 ?) images that are now augmented
    ## the_augmented_labels1 contains (4000*5) labels for the above images
    # the_test_images1 contains (1000) images
    # the_test_labels1 contains (1000) labels

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


    # (VERIFY SHAPE)
    # the_augmented_train1 contains (4000*5 ?) images that are now augmented
    ## the_augmented_labels1 contains (4000*5) labels for the above images
    # the_test_images1 contains (1000) images
    # the_test_labels1 contains (1000) labels

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


    # (VERIFY SHAPE)
    # the_augmented_train1 contains (4000*5 ?) images that are now augmented
    ## the_augmented_labels1 contains (4000*5) labels for the above images
    # the_test_images1 contains (1000) images
    # the_test_labels1 contains (1000) labels

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


    # (VERIFY SHAPE)
    # the_augmented_train1 contains (4000*5 ?) images that are now augmented
    ## the_augmented_labels1 contains (4000*5) labels for the above images
    # the_test_images1 contains (1000) images
    # the_test_labels1 contains (1000) labels


    X_train = [the_augmented_train1, the_augmented_train2, the_augmented_train3, the_augmented_train4, the_augmented_train5]
    X_test = [the_test_images1, the_test_images2, the_test_images3, the_test_images4, the_test_images5]
    y_train = [the_augmented_labels1, the_augmented_labels2, the_augmented_labels3, the_augmented_labels4, the_augmented_labels5 ]
    y_test = [the_test_labels1, the_test_labels2, the_test_labels3, the_test_labels4, the_test_labels5 ]


    
    


    # ____________________________________________________________________________________________________________________


    temp_list = list(zip(unity_test_images, unity_test_labels))
    random.shuffle(temp_list)
    f_list = list(zip(*temp_list))
    unity_test_images = np.asarray(list(f_list[0]))
    unity_test_labels = np.asarray(list(f_list[1]))

    # ____________________________________________________________________________________________________________________

    test_data_dict = {
        'images': unity_test_images,
        'labels': unity_test_labels
    }



    # ____________________________________________________________________________________________________________________
    # shuffle the contents of xs and ys while preseving the pairing

    temp_list2 = list(zip(xs, ys))
    random.shuffle(temp_list2)
    f_list2 = list(zip(*temp_list2))
    xs = np.asarray(list(f_list2[0]))
    ys = np.asarray(list(f_list2[1]))

    # ____________________________________________________________________________________________________________________

    training_data_dict = {
        'images': xs,
        'labels': ys
    }

    X_train = training_data_dict["images"]
    X_test = test_data_dict["images"]
    y_train = training_data_dict["labels"]
    y_test = test_data_dict["labels"]

    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_train_y = np_utils.to_categorical(encoded_Y)
    y_train = dummy_train_y

    encoder2 = LabelEncoder()
    encoder2.fit(y_test)
    encoded_Y2 = encoder2.transform(y_test)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_test_y = np_utils.to_categorical(encoded_Y2)
    y_test = dummy_test_y

    final_list = [X_train, X_test, y_train, y_test, og_split1, og_split2, og_split3, og_split4, og_split5]
    return final_list


def main():
    data_sets = load_training_and_test_data()


if __name__ == '__main__':
    main()

