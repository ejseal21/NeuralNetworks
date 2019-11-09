'''preprocess_data.py
Preprocessing data in STL-10 image dataset
Cole Turner and Ethan Seal
CS343: Neural Networks
Project 2: Multilayer Perceptrons
'''
import numpy as np


def preprocess_stl(imgs, labels):
    '''Preprocesses stl image data for training by a MLP neural network

    Parameters:
    ----------
    imgs: unint8 ndarray  [0, 255]. shape=(Num imgs, height, width, RGB color chans)

    Returns:
    ----------
    imgs: float64 ndarray [0, 1]. shape=(Num imgs N,)
    Labels: int ndarray. shape=(Num imgs N,). Contains int-coded class values 0,1,...,9

    TODO:
    1) Cast imgs to float64, normalize to the range [0,1]
    2) Flatten height, width, color chan dims. New shape will be (num imgs, height*width*chans)
    3) Compute the mean image across the dataset, subtract it from the dataset
    4) Fix class labeling. Should span 0, 1, ..., 9 NOT 1,2,...10
    '''
    print('imgs.shape',imgs.shape)


    imgs = imgs.astype(np.float64)
    # imgs = imgs - imgs.mean
    imgs = (imgs) / 255.0
    imgs = imgs - np.mean(imgs, axis = 0)
    # imgs = (imgs)/imgs.std()    
    imgs = np.reshape(imgs, (5000, -1))
    labels = labels - 1

    return imgs, labels
    


def create_splits(data, y, n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500):
    '''Divides the dataset up into train/test/validation/development "splits" (disjoint partitions)
    Parameters:
    ----------
    data: float64 ndarray. Image data. shape=(Num imgs, height*width*chans)
    y: ndarray. int-coded labels.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)

    TODO:
    1) Divvy up the images into train/test/validation/development non-overlapping subsets (see return vars)


    Train data shape:  (3500, 3072)
    Train labels shape:  (3500,)
    Test data shape:  (500, 3072)
    Test labels shape:  (500,)
    Validation data shape:  (500, 3072)
    Validation labels shape:  (500,)
    dev data shape:  (500, 3072)
    dev labels shape:  (500,)
    '''

    if n_train_samps + n_test_samps + n_valid_samps + n_dev_samps != len(data):
        samps = n_train_samps + n_test_samps + n_valid_samps + n_dev_samps
        print(f'Error! Num samples {samps} does not equal num images {len(data)}!')
        return

    # FILL IN CODE HERE
    print('data.shape',data.shape)

    #reshape the data into respective sizes and compress last 
    # x_train = data[0:int(len(data)*.7), :].reshape(n_train_samps, data.shape[1])
    x_train = data[:n_train_samps, :].reshape(n_train_samps, data.shape[1])
    
    y_train = y[:n_train_samps]
    x_test = data[n_train_samps:n_train_samps + n_test_samps, :].reshape(n_test_samps,data.shape[1])
    y_test = y[n_train_samps:n_train_samps + n_test_samps]
    x_val = data[n_train_samps + n_test_samps:n_train_samps + n_test_samps + n_valid_samps, :].reshape(n_valid_samps,data.shape[1])
    y_val = y[n_train_samps + n_test_samps:n_train_samps + n_test_samps + n_valid_samps]
    x_dev = data[n_train_samps + n_test_samps + n_valid_samps:, :].reshape(n_dev_samps,data.shape[1])
    y_dev = y[n_train_samps + n_test_samps + n_valid_samps:]


    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev
