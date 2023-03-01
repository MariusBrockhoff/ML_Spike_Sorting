"""
Inputs:
    raw_data = np.array containing raw data as produced by file_opener.py
    frequencies = List of frequencies used for low and high cutoff (or just low
    in case of highpass filter)
    fsample = sampling frequency in [Hz]
    filtering_method = Filtering method used, currently choose between 
    Butter_bandpass, Butter_highpass, Elliptic_bandpass or Elliptic_highpass. default: Butter_bandpass
    order = order of the filtering, default: 2
    
Outputs:
    filtered = np.array of filtered signal. Shape = (recorded data points, number of electrodes)
    
"""

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_high(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data,  axis=0) #filtfilt(b, a, data)
    return y

def butter_bandpass_filter_high(data, lowcut, fs, order=2):
    b, a = butter_bandpass_high(lowcut, fs, order=order)
    y = lfilter(b, a, data,  axis=0)
    return y


def ellip_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = ellip(order,0.1,40, [low, high], 'bandpass')
    #y = filtfilt(b, a, data)
    y = filtfilt(b, a, data,  axis=0, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    return y

def ellip_filter_high(data, lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = ellip(order,0.1,40, low, 'highpass')
    #y = filtfilt(b, a, data)
    y = filtfilt(b, a, data,  axis=0, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    return y


def filter_signal(raw_data, frequencies, fsample, filtering_method="Butter_bandpass", order=2):
    
    if filtering_method == "Butter_bandpass":
        lowcut = frequencies[0]
        highcut = frequencies[1]
    
        filtered = np.empty(shape=(raw_data.shape[0], raw_data.shape[1]))
        for i in range(raw_data.shape[1]):
          filtered[:,i] = butter_bandpass_filter(raw_data[:,i], lowcut, highcut, fsample=fsample, order=order)
        del recording_data
    
    elif filtering_method == "Butter_highpass":
        lowcut = frequencies[0]
    
        filtered = np.empty(shape=(raw_data.shape[0], raw_data.shape[1]))
        for i in range(raw_data.shape[1]):
          filtered[:,i] = butter_bandpass_filter_high(raw_data[:,i], lowcut, fsample=fsample, order=order)
        del recording_data
    
    elif filtering_method == "Elliptic_bandpass":
        lowcut = frequencies[0]
        highcut = frequencies[1]
    
        filtered = np.empty(shape=(raw_data.shape[0], raw_data.shape[1]))
        for i in range(raw_data.shape[1]):
          filtered[:,i] = ellip_filter(raw_data[:,i], lowcut, highcut, fsample=fsample, order=order)
        del recording_data
    
    elif filtering_method == "Elliptic_highpass":
        lowcut = frequencies[0]
    
        filtered = np.empty(shape=(raw_data.shape[0], raw_data.shape[1]))
        for i in range(raw_data.shape[1]):
          filtered[:,i] = ellip_filter_high(raw_data[:,i], lowcut, fsample=fsample, order=order)
        del recording_data
    
    else:
        raise ValueError("Please choose a valid filtering method! Chooose between Butter_bandpass, Butter_highpass, Elliptic_bandpass or Elliptic_highpass")
    
    return filtered