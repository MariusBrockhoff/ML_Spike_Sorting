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


def data_preparation(config, path_spike_file, data_prep_method, normalization, train_test_split, batch_size, benchmark=False):

    with open(path_spike_file, 'rb') as f:
        X = pickle.load(f)
        fsample = 20000
        labels = X[:, 0]
        spike_times = X[:, 1]
        spikes = X[:, 2:]
        del X

    if normalization == "MinMax":
        scaler = MinMaxScaler()

    elif normalization == "Standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Please specify valid data normalization method (MinMax, Standard)")

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
        raise ValueError("Please specify valied data preprocessing method (gradient, raw_spikes)")

    '''if config.MODEL_TYPE=='AttnAE_1' or config.MODEL_TYPE=='AttnAE_2':
        print('shape of data set before codons:', spikes.shape)

        # add start+stop codon
        spikes = np.insert(spikes, 0, -1, axis=1)
        print('shape of data set after start codon:', spikes.shape)

        spikes = np.append(spikes, np.array([[-1] for i in range(spikes.shape[0])]), axis=1)
        print('shape of data set after stop codon:', spikes.shape)'''

    if benchmark:
        split = train_test_split
        #k_sets = int(1/split)
        dataset_lst = []
        dataset_test_lst = []
        for j in range(config.BENCHMARK_START_IDX, config.BENCHMARK_END_IDX):
            number_of_test_samples = int(split*spikes.shape[0])
            x_test = spikes[number_of_test_samples*j:number_of_test_samples*(j+1), :]
            y_test = labels[number_of_test_samples*j:number_of_test_samples*(j+1)]
            x_train = np.delete(spikes, slice(number_of_test_samples*j, number_of_test_samples*(j+1)), 0)
            y_train = np.delete(labels, slice(number_of_test_samples*j, number_of_test_samples*(j+1)))

            config.EPOCH_SIZE = int(x_train.shape[0]/batch_size)
            dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=True).take(config.EPOCH_SIZE).shuffle(buffer_size=len(x_train))
            dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=True).take(config.EPOCH_SIZE).shuffle(buffer_size=len(x_test))

            dataset_lst.append(dataset)
            dataset_test_lst.append(dataset_test)
        return dataset_lst, dataset_test_lst

    else:
        split = train_test_split
        number_of_test_samples = int(split * spikes.shape[0])
        x_test = spikes[-number_of_test_samples:, :]
        y_test = labels[-number_of_test_samples:]
        x_train = spikes[:-number_of_test_samples, :]
        y_train = labels[:-number_of_test_samples]

        config.EPOCH_SIZE = int(x_train.shape[0] / batch_size)
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=True).take(
            config.EPOCH_SIZE).shuffle(buffer_size=len(x_train))
        dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=True).take(
            config.EPOCH_SIZE).shuffle(buffer_size=len(x_test))

        return dataset, dataset_test


