import pickle
import tensorflow as tf
import numpy as np

from scipy.fft import fft

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def gradient_transform(X, fsample):
    """
    Performs a gradient transform on the input data.

    This function calculates the gradient of the signal in the input data, which can be
    useful for certain types of data processing and feature extraction.

    Args:
        X (numpy.ndarray): The input data array.
        fsample (float): The sampling frequency of the input data.

    Returns:
        numpy.ndarray: The gradient transformed data.
    """
    time_step = 1 / fsample * 10**6  # Convert to microseconds
    X_grad = np.empty(shape=(X.shape[0], X.shape[1]-1, X.shape[2]))
    for i in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(X.shape[1]-1):
                X_grad[i, j, k] = (X[i, j+1, k] - X[i, j, k]) / time_step

    return X_grad
    
class MinMaxScaler3D(MinMaxScaler):

    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0], X.shape[1]*X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)

class StandardScaler3D(StandardScaler):

    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0], X.shape[1]*X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)


def data_preparation(model_config, data_preprocessing_config, pretraining_config, fintune_config, benchmark=False):
    """
        Prepares data for training machine learning models.

        This function loads and preprocesses data for training. It includes operations like
        shuffling, splitting, and normalizing data. It also sets up the data for benchmarking
        if required.

        Args:
            model_config: Configuration for the model.
            data_preprocessing_config: Configuration for data pre-processing.
            pretraining_config: Configuration for pretraining.
            fintune_config: Configuration for fine-tuning.
            benchmark (bool, optional): Flag to indicate whether benchmarking data is to be prepared. Default is False.

        Returns:
            Processed data ready for training or benchmarking.
        """
    batch_size = pretraining_config.BATCH_SIZE_NNCLR
    if benchmark:
        with open(pretraining_config.data_path, 'rb') as f:
            X = pickle.load(f)
            fsample = 20000
            
            spike_times = X[0]
            labels = X[1]
            spikes = X[2]
            print("spikes shape", spikes.shape)
            
            # Shuffle data
            rand_indices = np.random.permutation(spikes.shape[0])
            spikes = spikes[rand_indices]
            labels = labels[rand_indices]
            spike_times = spike_times[rand_indices]
            
            print("Ground Truth Number of classes:", len(np.unique(labels)))
            model_config.SEQ_LEN = spikes.shape[1]*spikes.shape[2]
            #model_config.SEQ_LEN = 256
            
            classes_from_data_pretrain = True
            classes_from_data_finetune = True
            if classes_from_data_pretrain:
                pretraining_config.N_CLUSTERS = len(np.unique(labels))
            if classes_from_data_finetune:
                fintune_config.PSEUDO_N_CLUSTERS = len(np.unique(labels))
            del X
            
            labels = np.concatenate((labels.reshape(len(labels),1),spike_times.reshape(len(spike_times),1)), axis=1) 

        if pretraining_config.DATA_NORMALIZATION == "MinMax":
            scaler = MinMaxScaler3D()

        elif pretraining_config.DATA_NORMALIZATION == "Standard":
            scaler = StandardScaler3D()
        else:
            raise ValueError("Please specify valid data normalization method (MinMax, Standard)")

        if pretraining_config.DATA_PREP_METHOD == "gradient":
            #spikes = gradient_transform(spikes, fsample)
            #print("spikes shape:", spikes.shape)
            spikes = scaler.fit_transform(spikes)
            print("after 3D minmax:", spikes.shape)
            #model_config.SEQ_LEN = model_config.SEQ_LEN - 1

        elif pretraining_config.DATA_PREP_METHOD == "raw_spikes":
            spikes = scaler.fit_transform(spikes)
            spikes = scaler.fit_transform(spikes)

        elif pretraining_config.DATA_PREP_METHOD == "fft":
            FT_spikes = np.abs(fft(spikes))[:, :33]
            spikes = scaler.fit_transform(FT_spikes)
            model_config.SEQ_LEN = 33

        else:
            raise ValueError("Please specify valid data preprocessing method (gradient, raw_spikes)")

        split = pretraining_config.TRAIN_TEST_SPLIT
        number_of_test_samples = int(split * spikes.shape[0])
        pretraining_config.QUEUE_SIZE = int(pretraining_config.QUEUE_SIZE * spikes.shape[0])
        x_test = spikes[-number_of_test_samples:, :]
        y_test = labels[-number_of_test_samples:,:]
        x_train = spikes[:-number_of_test_samples, :]
        y_train = labels[:-number_of_test_samples,:]

        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=False)
        dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=False)

    else:
        with open(pretraining_config.data_path, 'rb') as f:
            X = pickle.load(f)
            fsample = 20000
            
            spike_times = X[0]
            spikes = X[1]
            
            # Shuffle data
            rand_indices = np.random.permutation(spikes.shape[0])
            spikes = spikes[rand_indices]
            spike_times = spike_times[rand_indices]

        if pretraining_config.DATA_NORMALIZATION == "MinMax":
            scaler = MinMaxScaler3D()

        elif pretraining_config.DATA_NORMALIZATION == "Standard":
            scaler = StandardScaler()
        else:
            raise ValueError("Please specify valid data normalization method (MinMax, Standard)")

        if pretraining_config.DATA_PREP_METHOD == "gradient":
            #print("spikes shape:", spikes.shape)
            #grad_spikes = gradient_transform(spikes, fsample)
            #print("grad_spikes shape:", grad_spikes.shape)
            spikes = scaler.fit_transform(spikes)
            print("Data shape:", spikes.shape)

        elif pretraining_config.DATA_PREP_METHOD == "raw_spikes":
            spikes = scaler.fit_transform(spikes)
            spikes = scaler.fit_transform(spikes)

        elif pretraining_config.DATA_PREP_METHOD == "fft":
            FT_spikes = np.abs(fft(spikes))[:, :33]
            spikes = scaler.fit_transform(FT_spikes)
            model_config.SEQ_LEN = 33

        else:
            raise ValueError("Please specify valid data preprocessing method (gradient, raw_spikes)")

        split = pretraining_config.TRAIN_TEST_SPLIT
        number_of_test_samples = int(split * spikes.shape[0])
        pretraining_config.QUEUE_SIZE = int(pretraining_config.QUEUE_SIZE * spikes.shape[0])
        x_test = spikes[-number_of_test_samples:]
        y = spike_times #np.concatenate((electrode, spike_times), axis=1)
        y_test = y[-number_of_test_samples:]
        x_train = spikes[:-number_of_test_samples]
        y_train = y[:-number_of_test_samples]

        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=False)
        dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=False)



    return dataset, dataset_test, pretraining_config, fintune_config
