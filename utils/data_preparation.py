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


def data_preparation(model_config, pretraining_config, pretrain_method, benchmark=False):

    batch_size = pretraining_config.BATCH_SIZE if pretrain_method == "reconstruction" else pretraining_config.BATCH_SIZE_NNCLR

    with open(pretraining_config.DATA_SAVE_PATH, 'rb') as f:
        X = pickle.load(f)
        fsample = 20000
        labels = X[:, 0]
        spike_times = X[:, 1]
        spikes = X[:, 2:]
        del X

    if pretraining_config.DATA_NORMALIZATION == "MinMax":
        scaler = MinMaxScaler()

    elif pretraining_config.DATA_NORMALIZATION == "Standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Please specify valid data normalization method (MinMax, Standard)")

    if pretraining_config.DATA_PREP_METHOD == "gradient":
        grad_spikes = gradient_transform(spikes, fsample)
        spikes = scaler.fit_transform(grad_spikes)
        model_config.SEQ_LEN = 63
        
    elif pretraining_config.DATA_PREP_METHOD == "raw_spikes":
        spikes = scaler.fit_transform(spikes)
        spikes = scaler.fit_transform(spikes)
        model_config.SEQ_LEN = 64

    elif pretraining_config.DATA_PREP_METHOD == "fft":
        FT_spikes = np.abs(fft(spikes))[:, :33]
        spikes = scaler.fit_transform(FT_spikes)
        model_config.SEQ_LEN = 33

    else:
        raise ValueError("Please specify valid data preprocessing method (gradient, raw_spikes)")

    if benchmark:
        split = pretraining_config.TRAIN_TEST_SPLIT
        #k_sets = int(1/split)
        dataset_lst = []
        dataset_test_lst = []
        for j in range(pretraining_config.BENCHMARK_START_IDX, pretraining_config.BENCHMARK_END_IDX):
            number_of_test_samples = int(split*spikes.shape[0])
            x_test = spikes[number_of_test_samples*j:number_of_test_samples*(j+1), :]
            y_test = labels[number_of_test_samples*j:number_of_test_samples*(j+1)]
            x_train = np.delete(spikes, slice(number_of_test_samples*j, number_of_test_samples*(j+1)), 0)
            y_train = np.delete(labels, slice(number_of_test_samples*j, number_of_test_samples*(j+1)))

            epoch_size = int(x_train.shape[0]/batch_size)
            dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=True).take(epoch_size).shuffle(buffer_size=len(x_train))
            dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=True).take(epoch_size).shuffle(buffer_size=len(x_test))

            dataset_lst.append(dataset)
            dataset_test_lst.append(dataset_test)
        return dataset_lst, dataset_test_lst

    else:
        split = pretraining_config.TRAIN_TEST_SPLIT
        number_of_test_samples = int(split * spikes.shape[0])
        x_test = spikes[-number_of_test_samples:, :]
        y_test = labels[-number_of_test_samples:]
        x_train = spikes[:-number_of_test_samples, :]
        y_train = labels[:-number_of_test_samples]

        epoch_size = int(x_train.shape[0] / batch_size)
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=True).take(
            epoch_size).shuffle(buffer_size=len(x_train))
        dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=True).take(
            epoch_size).shuffle(buffer_size=len(x_test))

        return dataset, dataset_test


