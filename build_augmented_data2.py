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

    # ______________________________________________________________________________________
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

    # for every image x, augmentations are performed on it and the resulting images are stored in the savedir
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
                break'''

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
        cv2.imwrite(os.path.join(savedir, unique + 'v.jpg'), xvflip)

            # ______________________________________________________________________________________
    def norm_image(image):
        return (image - np.mean(image)) / np.std(image)

    # ______________________________________________________________________________________
    grand_test_final = []
    grand_test_labels = []
    shutil.rmtree("colorectal-histology-mnist/Kather_texture_2016_image_tiles_28000/")
    os.mkdir("colorectal-histology-mnist/Kather_texture_2016_image_tiles_28000/")
    for the_fi in os.listdir(new_dirr):  # iterate over each label 01_TUMOR .... 08_EMPTY
        os.mkdir("colorectal-histology-mnist/Kather_texture_2016_image_tiles_28000/" + the_fi)
        star_dir = new_dirr + the_fi  # make note of the directory that now includes this folder
        the_list = []
        for filename in os.listdir(star_dir):
            the_list.append(filename)  # make the_list contain string names of all the images in the label folder

        random.shuffle(the_list)  # shuffle the image names
        og_split1 = the_list[:125]  # split into 5 categories
        og_split2 = the_list[125:250]  # split into 5 categories
        og_split3 = the_list[250:375]  # split into 5 categories
        og_split4 = the_list[375:500]  # split into 5 categories
        og_split5 = the_list[500:]  # split into 5 categories

        og_train = the_list[:500]  # slice first 500 image names as training images
        og_test = the_list[500:]  # slice last 125 image names as testing images

        og_test_final = str_to_img(star_dir, og_test)  # list of numpy array of 125 test images
        og_test_labels = gen_labels(the_fi, og_test)  # list of string labels for 125 test images

        grand_test_final.append(og_test_final)
        grand_test_labels.append(og_test_labels)

        for filename2 in og_train:
            temp_dir = star_dir + "/" + filename2
            augment5(temp_dir, "colorectal-histology-mnist/Kather_texture_2016_image_tiles_28000/" + the_fi)

        #rename_for_folder("colorectal-histology-mnist/Kather_texture_2016_image_tiles_28000/" + the_fi, the_fi + "_")

    unity_test_images = np.asarray(grand_test_final)  # array conversion
    unity_test_labels = np.asarray(grand_test_labels)

    unity_test_images = np.concatenate([unity_test_images])
    unity_test_labels = np.concatenate([unity_test_labels])


    # ____________________________________________________________________________________________________________________
    # shuffle the contents of unity_test_images and unity_test_labels while preseving the pairing
    unity_test_images = unity_test_images.reshape((1000, 150, 150, 3))
    unity_test_labels = unity_test_labels.flatten()


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
    X, Y = load_batch("colorectal-histology-mnist/Kather_texture_2016_image_tiles_28000")

    xs = X
    ys = Y
    xs = np.concatenate([xs])
    ys = np.concatenate([ys])

    for i in range(len(xs)):
        xs[i] = norm_image(xs[i])

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

    final_list = [X_train, X_test, y_train, y_test, og_split1,og_split2, og_split3, og_split4, og_split5]
    return final_list



def main():
    data_sets = load_training_and_test_data()


if __name__ == '__main__':
    main()

