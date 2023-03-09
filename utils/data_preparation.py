# -*- coding: utf-8 -*-
"""
Load Spike data and prepare for training of model
"""
import pickle
import tensorflow as tf
import numpy as np

from scipy.fft import fft

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def gradient_transform(X, fsample):
  time_step = 1 / fsample * 10**6 #in us
  X_grad = np.empty(shape=(X.shape[0], X.shape[1]-1))
  for i in range(X.shape[0]):
    for j in range(X.shape[1]-1):
      X_grad[i, j] = (X[i,j+1] - X[i,j])/time_step

  return X_grad


def data_preparation(config, path_spike_file, data_prep_method, normalization, train_test_split, batch_size):

    print('---' * 30)
    print('IMPORTING DATASET...')
    with open(path_spike_file, 'rb') as f:
        X = pickle.load(f)
        fsample = 20000
        y_train = X[:, 0]
        spike_times = X[:, 1]
        spikes = X[:, 2:]
        del X

    print('---' * 30)
    print('DATA PREPARATION...')

    if normalization == "MinMax":
        scaler = MinMaxScaler()

    elif normalization == "Standard":
        scaler = StandardScaler()

    if data_prep_method == "gradient":
        grad_spikes = gradient_transform(spikes, fsample)
        spikes = scaler.fit_transform(grad_spikes)
        config.SEQ_LEN = 63
        
    elif data_prep_method == "raw_spikes":
        spikes = scaler.fit_transform(spikes)
        spikes = scaler.fit_transform(spikes)
        config.SEQ_LEN = 64

    elif data_prep_method == "fft":
        FT_spikes = np.abs(fft(spikes))[:, :33]
        spikes = scaler.fit_transform(FT_spikes)
        config.SEQ_LEN = 33

    else:
        print("Please specify valied data preprocessing method (gradient, raw_spikes)")

    '''if config.MODEL_TYPE=='AttnAE_1' or config.MODEL_TYPE=='AttnAE_2':
        print('shape of data set before codons:', spikes.shape)

        # add start+stop codon
        spikes = np.insert(spikes, 0, -1, axis=1)
        print('shape of data set after start codon:', spikes.shape)

        spikes = np.append(spikes, np.array([[-1] for i in range(spikes.shape[0])]), axis=1)
        print('shape of data set after stop codon:', spikes.shape)'''

    print('---' * 30)
    print('DATA SPLIT...')
    split = train_test_split
    number_of_test_samples = int(split*spikes.shape[0])
    x_test = spikes[-number_of_test_samples:,:]
    y_test = y_train[-number_of_test_samples:]
    x_train = spikes[:-number_of_test_samples,:]
    y_train = y_train[:-number_of_test_samples]
    
    print("Shapes data:")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    print('---' * 30)
    print('CREATE BATCHES...')
    config.EPOCH_SIZE = int(x_train.shape[0]/batch_size)
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder=True).take(config.EPOCH_SIZE).shuffle(buffer_size=len(x_train))
    dataset_test = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size, drop_remainder=True).take(config.EPOCH_SIZE).shuffle(buffer_size=len(x_test))
    
    return dataset, dataset_test


def make_batches(data, batch_size, shuffle, preprocessing_method):
    if shuffle:
        data = tf.random.shuffle(data)

    mod = data.shape[0] % batch_size

    if data.shape[0] % batch_size != 0:
        data = data[:-mod]

    batch_number = int(data.shape[0] / batch_size)

    if preprocessing_method != 0:
        data = tf.reshape(data, shape=(batch_number, batch_size, data.shape[-2], data.shape[-1]))

    else:
        data = tf.reshape(data, shape=(batch_number, batch_size, data.shape[-1]))

    return data

def data_preprocessing_Jakob(dataset_path, method, data_split):
    # file_name: e.g. 'C_Difficult1_noise02' with default path '/content/drive/My Drive/SpikeSorting_Datasets/file_name/.mat
    # method: ['gradient', 'FT'] --> gradient + MinMaxScale, FourierTransform
    # data_split: boolean for split in train and test data

    # Import dataset
    print('---' * 30)
    print('IMPORTING DATASET...')
    with open(dataset_path, 'rb') as f:
        raw_data = pickle.load(f)

        spike_times = raw_data[:, 1]
        spike_class = raw_data[:, 0]
        X = raw_data[:, 2:]

    print('shape of raw data:', raw_data.shape)
    print('shape of spikes:', X.shape)

    # Data preparation:
    print('---' * 30)
    print('DATA PREPARATION...')
    # method 1: Transform to gradient and then MinMaxScale
    if method == 'gradient':
        signal_length = 65
        X_grad = gradient_transform(X, fsample=20000)
        scaler = MinMaxScaler()
        X_data = scaler.fit_transform(X_grad)

    # method 2: FT
    elif method == 'FT':
        signal_length = 35
        X_data = np.abs(fft(X))[:,:33]  # due to real-value sequence --> FT-spectrum is symmetric (1 31 1 31) take 33 unique of inital 64 values
        scaler = MinMaxScaler()
        X_data = scaler.fit_transform(X_data)

    # append start+stop
    if method == 'gradient' or method == 'FT':
        print('shape of data set (using gradient and MinMaxScale) before codons:', X_data.shape)

        # add start+stop codon
        X_data = np.insert(X_data, 0, -1, axis=1)
        print('shape of data set (using gradient and MinMaxScale) after start codon:', X_data.shape)

        X_data = np.append(X_data, np.array([[-1] for i in range(X_data.shape[0])]), axis=1)
        print('shape of data set (using gradient and MinMaxScale) after stop codon:', X_data.shape)

    # create batches
    print('---' * 30)
    print('CREATE BATCHES...')
    print('shape of data before splitting into batches', X_data.shape)

    batches = make_batches(X_data, batch_size=64, shuffle=False, preprocessing_method=0)
    print('shape of data after splitting into batches', batches.shape)

    # train/test split
    if data_split:
        print('---' * 30)
        print('DATA SPLIT...')
        split_frac = 0.1

        split_ind = int(batches.shape[0] * split_frac)

        batches_train = batches[split_ind:]
        batches_test = batches[:split_ind]

        print('shape of batches_train', batches_train.shape)
        print('shape of batches_test', batches_test.shape)
    print('---' * 30)

    return batches_train, batches_test, signal_length