# -*- coding: utf-8 -*-

'''
Helper methods to read AFAD dataset from disk,
convert to tensor and write tensor to hdf5 file.
Because our training set is quite big,
we perform training on hdf5 file to minimize disk read.
'''

import json
import os
from math import floor

import cv2
import numpy as np
import progressbar
from imutils import paths
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from hdf5_helper.hdf5_writer import HDF5Writer


def split_by_gender(data_dir):
    '''
    Label each image as either male or female.
    All female images are stored in folder named "112",
    while male images are stored in folder named "111".

    Returns: a tuple of 2 arrays of the same length.
    The first array is the path to each image,
    while the second array is the corresponding label (gender).
    '''

    data_paths = list(paths.list_images(data_dir))
    labels = ['0' if n.split(os.path.sep)[-2] == '111' else '1' for n in data_paths]

    if len(data_paths) == 0:
        return None, None

    # Because folder name is string, we need to encode it before using it as label.
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    return data_paths, labels

def split_by_age(data_dir):
    '''
    Label each image by age.
    The folder where each image is stored is the age.

    Returns: a tuple of 2 arrays of the same length.
    The first array is the path to each image,
    while the second array is the corresponding label (age).
    '''

    data_paths = list(paths.list_images(data_dir))

    if len(data_paths) == 0:
        return None, None

    labels = [int(n.split(os.path.sep)[-3]) for n in data_paths]

    return data_paths, labels

def write_data_to_hdf5(data_dir, split_method, output_dir, set_split=0.2, channels_first=False):
    data_paths, labels = split_method(data_dir)

    if data_paths is None:
        print('Cannot find any image in {0}'.format(data_dir))
        return

    test_size = floor(len(data_paths) * set_split)
    train_val_paths, test_paths, train_val_labels, test_lables = train_test_split(
        data_paths, labels, test_size=test_size)

    val_size = floor(len(train_val_paths) * set_split)
    train_paths, val_paths, train_labels, val_lables = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size)

    write_set_to_hdf5(train_paths, train_labels, output_dir, 'training.hdf5', channels_first=channels_first)
    write_set_to_hdf5(val_paths, val_lables, output_dir, 'validation.hdf5', channels_first=channels_first)
    write_set_to_hdf5(test_paths, test_lables, output_dir, 'test.hdf5', channels_first=channels_first)

def write_set_to_hdf5(data_paths, labels, output_dir, output_name, channels_first=False):
    '''
    Read images data, convert to tensor and write both tensor data and label to one hdf5 file.
    Image can be either channels first or channels_last. 
    '''

    print('Writing to {0}...'.format(output_name))

    dims = (len(data_paths), 3, 150, 150) if channels_first else (len(data_paths), 150, 150, 3)
    writer = HDF5Writer(dims, output_dir, output_name)

    # Progress bar to track writing process
    widgets = ['Processing:', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(data_paths), widgets=widgets).start()

    R, G, B = [], [], [] # save rgb values for mean normalization, only for training set
    images_buffer, labels_buffer = [], []

    # For each path in image path's list
    for i in np.arange(len(data_paths)):
        # Read image, resize to 150x150 pixel
        image = load_img(
            path = data_paths[i],
            grayscale=False,
            color_mode='rgb',
            target_size=(150,150),
            interpolation='nearest'
        )

        # Convert image to tensor
        data_format = 'channels_first' if channels_first else 'channels_last'
        image = img_to_array(image, data_format=data_format)

        # Append current mean R, G, B values to array
        if output_name == 'training.hdf5':
            axis = (1, 2) if channels_first else (0, 1) 
            r, g, b = image.mean(axis=axis)
            R.append(r)
            G.append(g)
            B.append(b)
        
        images_buffer.append(image)
        labels_buffer.append(labels[i])

        # If buffer is full, write all data in buffer to hdf5 file
        if (i + 1) % writer.buffer_size == 0:
            writer.write(images_buffer, labels_buffer)
            images_buffer, labels_buffer = [], []

        pbar.update(i) # update progress bar

    # If there is any data left in buffer, flush it to hdf5 file
    if len(images_buffer) > 0:
        writer.write(images_buffer, labels_buffer)

    writer.close()

    # For training set, calculate to mean values of R, G, B channels and write to json file
    if output_name == 'training.hdf5':
        print('Calculating color means...')
        mean = {'R': np.mean(R).astype('float'), 'G': np.mean(G).astype('float'), 'B': np.mean(B).astype('float')} # cannot dumps float32

        mean_path = os.path.join(output_dir, 'mean.json')
        with open(mean_path, 'w') as f:
            f.write(json.dumps(mean))

    pbar.finish() # update progress bar
