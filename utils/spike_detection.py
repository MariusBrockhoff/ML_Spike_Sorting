# -*- coding: utf-8 -*-
import numpy as np
import math
import pickle


def spike_detection(filtered_signal, electrode_stream, fsample, save_path, min_TH, dat_points_pre_min=20,
                    dat_points_post_min=44, max_TH=-30, chunck_len=300, refrec_period=0.002, reject_channels=[None],
                    file_name=None):
    """
    Detects spikes in a filtered signal and saves the results.

    This function processes a filtered signal to detect spikes based on threshold crossing and other criteria.
    It divides the signal into chunks and scans for local minima that exceed specified thresholds. Identified spikes
    are saved in a specified directory.

    Parameters:
    filtered_signal (numpy.ndarray): The filtered signal data.
    electrode_stream: The stream of electrode data.
    fsample (int): The sampling rate of the signal.
    save_path (str): The path to save spike detection results.
    min_TH (float): The minimum threshold for spike detection.
    dat_points_pre_min (int, optional): The number of data points before the minimum to consider for a spike. Default
                                        is 20.
    dat_points_post_min (int, optional): The number of data points after the minimum to consider for a spike. Default is
                                         44.
    max_TH (float, optional): The maximum threshold for spike detection. Default is -30.
    chunck_len (int, optional): The length of each signal chunk for processing. Default is 300.
    refrec_period (float, optional): The refractory period to avoid multiple detections of the same spike. Default is
    0.002 seconds.
    reject_channels (list, optional): List of channels to reject from spike detection. Default is [None].
    file_name (str, optional): The name of the file to save spike detection results. If None, a default name is used.

    The function scans the filtered signal, detects spikes based on the thresholds, and stores the detected spikes and
    their metadata. The results are saved in a pickle file at the specified 'save_path'. If 'file_name' is provided, it
    is used in the saved file name.

    Returns:
    dict: A dictionary containing spike detection results and metadata.
    """

    reject_ch = reject_channels  # In case you manually want to reject channels
    ids = [c.channel_id for c in electrode_stream.channel_infos.values()]
    length_of_chunck = chunck_len  # in
    points_pre = dat_points_pre_min
    points_post = dat_points_post_min
    overlap = 20  # 10 data point overlap between chunks to check whetehr it is the local minima.
    t = 0  # rejects the 1st second to avoid ripples in filter

    # finding the number of chunks. (e.g. is recoring is 60.5s, no_chunks = 61)
    # no_chunks = 20: for specifying a time window manually
    no_chunks = math.ceil(filtered_signal.shape[
                              0] / fsample / length_of_chunck)
    spk = list()
    spike_times = [[0] for x in range(filtered_signal.shape[1])]

    while t < no_chunks:  # rejects the incomplete chunk at the end to avoid filter ripple
        if (t + 1) * fsample * length_of_chunck <= len(filtered_signal):
            chunk = filtered_signal[t * fsample * length_of_chunck:((t + 1) * fsample * length_of_chunck + overlap - 1)]
            # print("complete chunk")
        else:
            chunk = filtered_signal[t * fsample * length_of_chunck:]
            # stdev = np.std(chunk, axis = 0)
        med = np.median(np.absolute(chunk) / 0.6745, axis=0)

        for index in range(len(chunk)):
            if points_pre < index < fsample * length_of_chunck - points_post:
                # choose the threshold value. Finds which channels exceed the threshold at this instance
                threshold_cross = chunk[index, :] < min_TH * med
                threshold_arti = chunk[index, :] > max_TH * med

                threshold = threshold_cross * threshold_arti
                # *stim_reject#finds out which electrode crosses -5*SD and ignores stimulation artefacts
                probable_spike = threshold

                if np.sum(probable_spike > 0):
                    for e in range(filtered_signal.shape[1]):
                        channel_id = ids[e]
                        channel_info = electrode_stream.channel_infos[channel_id]
                        ch = int(channel_info.info['Label'][-2:])
                        if probable_spike[e] == 1 and not (
                                ch in reject_ch):  # whether threshold exceeded at an electrode and if it is rejected
                            t_diff = (fsample * t * length_of_chunck + index) - spike_times[e][-1]
                            if t_diff > refrec_period * fsample and chunk[index, e] == np.min(
                                    chunk[(index - points_pre):(index + points_post), e]):
                                # whether the spike is 2ms apart and whether it is the true minumum
                                # and not just any point below -5*SD
                                spike_times[e].append(fsample * t * length_of_chunck + index)
                                if (fsample * t * length_of_chunck + index + points_post) < filtered_signal.shape[0]:
                                    # making sure that the whole spike waveform is within the limits of the
                                    # filtered signal array
                                    spk_wave = list(filtered_signal[
                                                    (fsample * t * length_of_chunck + index - points_pre):(
                                                                fsample * t * length_of_chunck + index + points_post),
                                                    e])
                                    # selecting 1.6ms around the spike time from the whole fltered signal array
                                    spk_wave.insert(0, (fsample * t * length_of_chunck + index))
                                    spk_wave.insert(0, ch)
                                    spk.append(spk_wave)

        t = t + 1

    print("Total number of detected spikes:", len(spk))
    dat_arr = np.array(spk)
    results = {"Filename": file_name, "Sampling rate": fsample, "Recording len": filtered_signal.shape[0] / fsample,
               "Raw_spikes": dat_arr[dat_arr[:, 1].argsort()]}

    if file_name is not None:
        with open(save_path + '/Spike_File_' + file_name + '.pkl', 'wb+') as f:
            pickle.dump(results, f, -1)
    else:
        with open(save_path + '/Spike_File.pkl', 'wb+') as f:
            pickle.dump(results, f, -1)

    return results
