# -*- coding: utf-8 -*-
"""
Load Spike data and prepare for training of model
"""
import pickle
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
 

def gradient_transform(X, fsample):
  time_step = 1 / fsample * 10**6 #in us
  X_grad = np.empty(shape=(X.shape[0], X.shape[1]-1))
  for i in range(X.shape[0]):
    for j in range(X.shape[1]-1):
      X_grad[i, j] = (X[i,j+1] - X[i,j])/time_step

  return X_grad


def data_praparation(path_spike_file, data_prep_method, normalization, train_test_split, batch_size):
    
    with open(path_spike_file, 'rb') as f:
        X = pickle.load(f)
        spikes = X["Raw_spikes"]
        #recording_len = X["Recording len"]
        fsample = X["Sampling rate"]
        del X
    
    #spike_times = spikes[:,1]
    #electrodes = spikes[:,0]
    spikes = spikes[:,2:]
    
    if data_prep_method == "gradient":
        grad_spikes = gradient_transform(spikes, fsample)
        if normalization == "MinMax":
            scaler = MinMaxScaler()
        elif normalization == "Standard":
            scaler = StandardScaler()
        x_train = scaler.fit_transform(grad_spikes)
        config.SEQ_LEN = 63
        
    elif data_prep_method == "raw_spikes":
        if normalization == "MinMax":
            scaler = MinMaxScaler()
        elif normalization == "Standard":
            scaler = StandardScaler()
        x_train = scaler.fit_transform(spikes)
        
    else:
        print("Please specify valied data preprocessing method (gradient, raw_spikes)")


    split = train_test_split
    number_of_test_samples = int(split*x_train.shape[0])
    x_test = x_train[-number_of_test_samples:,:]
    #y_test = y_train[-number_of_test_samples:]
    x_train = x_train[:-number_of_test_samples,:]
    #y_train = y_train[:-number_of_test_samples] 
    
    print("Shapes data:")
    print("x_train:", x_train.shape)
    #print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    #print("y_test:", y_test.shape)
    
    
    EPOCH_SIZE = int(x_train.shape[0]/batch_size)
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder=True).take(EPOCH_SIZE).shuffle(buffer_size=len(x_train))
    dataset_test = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size, drop_remainder=True).take(EPOCH_SIZE).shuffle(buffer_size=len(x_test))
    
    return dataset, dataset_test
