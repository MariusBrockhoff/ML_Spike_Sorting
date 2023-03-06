import pickle

path_spike_file = '/ML_Spike_Sorting/Data/'

with open(path_spike_file, 'rb') as f:
    X = pickle.load(f)
    print('shapes of Data', X.shape)
    spikes = X["Raw_spikes"]
    print('shapes of spikes', spikes.shape)
    # recording_len = X["Recording len"]
    fsample = X["Sampling rate"]
    del X