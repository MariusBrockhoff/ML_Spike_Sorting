# -*- coding: utf-8 -*-
"""
Inputs:
    filtered_signal = np.array of filtered signal. Shape = (recorded data points, number of electrodes)
    as produced by filter_signal.py
    electrode_stream = original data file containing all raw data and channel infos
    as produced by filter_signal.py
    fsample = sampling frequency in [Hz]
    save_path = path to save a pickle file containing the detected spike file (if save==True)
    min_TH = Minimum threshold for detecting spikes that is multiplied with noise estimation (usually -5)
    dat_points_pre_min = Number of data points extracted for the spike shapes before the minimum. default: 20
    dat_points_post_min= Number of data points extracted for the spike shapes after the minimum. default: 44
    max_TH = Maximum threshold for detecting spikes that is multiplied with noise estimation.
    This is to filter massive signals that are more likely artefacts than spikes. default: -30
    chunck_len = Length in seconds in which the trace is cut while detection to minimise RAM usage.
    Can be set to recording length if all should be done at once. default: 300
    refrec_period = Assumed refractory period of a neuron in seconds which we "block" between
    spikes to minimuse overlapping spikes. default: 0.002 
    reject_channels = List of channels/electrodes that should not be included in
    the spike detection. default: [None]
    file_name = Name of the analysed file for reference, string. default: None
    
Outputs:
    Results = Dictionary containing all detected spikes, a file name, the sampling 
    frequency and recording length. Spikes are stored in key Raw_spikes, which contains  
    a np.array of shape (number of spikes x 2 + dat_points_pre_min + dat_points_post_min)
    First column contains the electrode at which the spike was recorded, the second column
    contains the spike time and the rest of columns contain the spike shape
"""

def spike_detection(filtered_signal, electrode_stream, fsample, save_path, min_TH, dat_points_pre_min=20, dat_points_post_min=44, max_TH=-30, chunck_len=300, refrec_period=0.002, reject_channels=[None], file_name=None):
    
    reject_ch = reject_channels #In case you manually want to reject channels
    ids = [c.channel_id for c in electrode_stream.channel_infos.values()]
    length_of_chunck = chunck_len #in  
    points_pre = dat_points_pre_min
    points_post = dat_points_post_min
    overlap = 20 #10 data point overlap between chunks to check whetehr it is the local minima.
    t = 0 #rejects the 1st second to avoid ripples in filter
    
    no_chunks = math.ceil(filtered_signal.shape[0]/fsample/length_of_chunck) #finding the number of chunks. (e.g. is recoring is 60.5s, no_chunks = 61)#no_chunks = 20 #for specifying a time window manually
    spk = list()
    spike_times = [[0] for x in range(filtered_signal.shape[1])]
    
    
    while t < no_chunks: #rejects the incomplete chunk at the end to avoid filter ripple
      if (t+1)*fsample*length_of_chunck <= len(filtered_signal):
        chunk = filtered_signal[t*fsample*length_of_chunck:((t+1)*fsample*length_of_chunck + overlap - 1)] 
          #print("complete chunk")
      else:
        chunk = filtered_signal[t*fsample*length_of_chunck:] 
      #stdev = np.std(chunk, axis = 0)
      med = np.median(np.absolute(chunk)/0.6745, axis=0) 
    
      for index in range(len(chunk)):
        if index > points_pre and index <fsample*length_of_chunck-points_post:
          threshold_cross = chunk[index, :] < min_TH*med  #choose the threshold value. Finds which channels exceed the threshold at this instance
          threshold_arti =  chunk[index, :] > max_TH*med
    
          threshold = threshold_cross*threshold_arti
          probable_spike = threshold #*stim_reject#finds out which electrode crosses -5*SD and ignores stimulation artefacts
            
          if np.sum(probable_spike > 0):
            for e in range(filtered_signal.shape[1]):
              channel_id = ids[e]
              channel_info = electrode_stream.channel_infos[channel_id]
              ch = int(channel_info.info['Label'][-2:])
              if probable_spike[e] == 1 and not(ch in reject_ch): #whether threshold exceeded at an electrode and if it is rejected
                t_diff = (fsample*t*length_of_chunck + index) - spike_times[e][-1]
                if t_diff > refrec_period*fsample and chunk[index, e] == np.min(chunk[(index-points_pre):(index+points_post), e]): #whether the spike is 2ms apart and whether it is the true minumum and not just any point below -5*SD
                  spike_times[e].append(fsample*t*length_of_chunck + index)
                  if (fsample*t*length_of_chunck + index + points_post) < filtered_signal.shape[0]: #making sure that the whole spike waveform is within the limits of the filtered signal array
                    spk_wave = list(filtered_signal[(fsample*t*length_of_chunck + index - points_pre):(fsample*t*length_of_chunck + index + points_post), e])#selecting 1.6ms around the spike time from the whole fltered signal array
                    spk_wave.insert(0, (fsample*t*length_of_chunck + index))
                    spk_wave.insert(0, ch)
                    spk.append(spk_wave)
        
    
      t = t+1

    print("Total number of detected spikes:", len(spk))    
    dat_arr = np.array(spk)
    Results = {}
    Results["Filename"] = file_name
    Results["Sampling rate"] = fsample
    Results["Recording len"] = filtered_signal.shape[0] / fsample
    Results["Raw_spikes"] = dat_arr[dat_arr[:, 1].argsort()]
    
    if file_name != None:
        with open(save_path + '/Spike_File_' + file_name + '.pkl', 'wb+') as f:
            pickle.dump(Results, f, -1)
    else:
        with open(save_path + '/Spike_File.pkl', 'wb+') as f:
            pickle.dump(Results, f, -1)
    
    return Results
    
