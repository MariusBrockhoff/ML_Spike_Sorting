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


def data_preparation(model_config, pretraining_config, pretrain_method, fintune_config, benchmark=False):

    batch_size = pretraining_config.BATCH_SIZE if pretrain_method == "reconstruction" else pretraining_config.BATCH_SIZE_NNCLR

    with open(pretraining_config.DATA_SAVE_PATH, 'rb') as f:
        X = pickle.load(f)
        fsample = 20000
        print("X shape", X.shape)
        #Shuffle data
        np.random.shuffle(X)
        labels = X[:, 0]
        print("Ground Truth Number of classes:", len(np.unique(labels)))
        spike_times = X[:, 1]
        spikes = X[:, 2:]
        model_config.SEQ_LEN = spikes.shape[1]
        classes_from_data_pretrain = True
        classes_from_data_finetune = True
        pretraining_config.QUEUE_SIZE = int(pretraining_config.QUEUE_SIZE * X.shape[0])
        if classes_from_data_pretrain:
            pretraining_config.N_CLUSTERS = len(np.unique(labels))
        if classes_from_data_finetune:
            fintune_config.DEC_N_CLUSTERS = len(np.unique(labels))
            fintune_config.IDEC_N_CLUSTERS = len(np.unique(labels))
            fintune_config.PSEUDO_N_CLUSTERS = len(np.unique(labels))
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
        model_config.SEQ_LEN = model_config.SEQ_LEN - 1
        
    elif pretraining_config.DATA_PREP_METHOD == "raw_spikes":
        spikes = scaler.fit_transform(spikes)
        spikes = scaler.fit_transform(spikes)

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

            dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=True)
            dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=True)

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


        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=True)
        dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=True)

        return dataset, dataset_test, pretraining_config, fintune_config


