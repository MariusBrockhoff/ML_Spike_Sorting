"""
Module for filtering raw signal data using different filtering methods.

This module contains functions to apply Butterworth and Elliptic filters, both bandpass
and highpass, to raw signal data. It supports filtering data for individual channels and
allows for specification of filter order and cutoff frequencies.

Functions:
    butter_bandpass(lowcut, highcut, fs, order=2): Creates a Butterworth bandpass filter.
    butter_bandpass_high(lowcut, fs, order=5): Creates a Butterworth highpass filter.
    butter_bandpass_filter(data, lowcut, highcut, fs, order=2): Applies a Butterworth bandpass filter to the data.
    butter_bandpass_filter_high(data, lowcut, fs, order=2): Applies a Butterworth highpass filter to the data.
    ellip_filter(data, lowcut, highcut, fs, order=5): Applies an Elliptic bandpass filter to the data.
    ellip_filter_high(data, lowcut, fs, order=5): Applies an Elliptic highpass filter to the data.
    filter_signal(raw_data, frequencies, fsample, filtering_method="Butter_bandpass", order=2): Main function to filter
    the signal based on specified method and parameters.
"""


import numpy as np
from scipy.signal import butter, lfilter, filtfilt, ellip


def butter_bandpass(lowcut, highcut, fs, order=2):
    """
    Create coefficients for a Butterworth bandpass filter.

    Args:
        lowcut (float): The low cutoff frequency.
        highcut (float): The high cutoff frequency.
        fs (float): The sampling frequency.
        order (int, optional): The order of the filter. Default is 2.

    Returns:
        tuple: Filter coefficients (b, a).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_high(lowcut, fs, order=5):
    """
    Create coefficients for a Butterworth highpass filter.

    Args:
        lowcut (float): The low cutoff frequency.
        fs (float): The sampling frequency.
        order (int, optional): The order of the filter. Default is 5.

    Returns:
        tuple: Filter coefficients (b, a).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Apply a Butterworth bandpass filter to the data.

    Args:
        data (numpy.ndarray): The data to filter.
        lowcut (float): The low cutoff frequency.
        highcut (float): The high cutoff frequency.
        fs (float): The sampling frequency.
        order (int, optional): The order of the filter. Default is 2.

    Returns:
        numpy.ndarray: The filtered data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data,  axis=0)  # filtfilt(b, a, data)
    return y


def butter_bandpass_filter_high(data, lowcut, fs, order=2):
    """
    Apply a Butterworth highpass filter to the data.

    Args:
        data (numpy.ndarray): The data to filter.
        lowcut (float): The low cutoff frequency.
        fs (float): The sampling frequency.
        order (int, optional): The order of the filter. Default is 2.

    Returns:
        numpy.ndarray: The filtered data.
    """
    b, a = butter_bandpass_high(lowcut, fs, order=order)
    y = lfilter(b, a, data,  axis=0)
    return y


def ellip_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply an Elliptic bandpass filter to the data.

    Args:
        data (numpy.ndarray): The data to filter.
        lowcut (float): The low cutoff frequency.
        highcut (float): The high cutoff frequency.
        fs (float): The sampling frequency.
        order (int, optional): The order of the filter. Default is 5.

    Returns:
        numpy.ndarray: The filtered data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = ellip(order, 0.1, 40, [low, high], 'bandpass')
    # y = filtfilt(b, a, data)
    y = filtfilt(b, a, data,  axis=0, padtype='odd', padlen=3*(max(len(b), len(a))-1))
    return y


def ellip_filter_high(data, lowcut, fs, order=5):
    """
    Apply an Elliptic highpass filter to the data.

    Args:
        data (numpy.ndarray): The data to filter.
        lowcut (float): The low cutoff frequency.
        fs (float): The sampling frequency.
        order (int, optional): The order of the filter. Default is 5.

    Returns:
        numpy.ndarray: The filtered data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = ellip(order, 0.1, 40, low, 'highpass')
    # y = filtfilt(b, a, data)
    y = filtfilt(b, a, data,  axis=0, padtype='odd', padlen=3*(max(len(b), len(a))-1))
    return y


def filter_signal(raw_data, frequencies, fsample, filtering_method="Butter_bandpass", order=2):
    """
    Filter the signal based on specified method and parameters.
    This function applies the selected filtering method (Butterworth or Elliptic, bandpass or highpass)
    to the provided raw data. It can handle multiple channels in the data.

    Args:
        raw_data (numpy.ndarray): The raw data to filter.
        frequencies (list): The list of frequencies used for low and high cutoff (or just low for highpass filter).
        fsample (float): The sampling frequency in Hz.
        filtering_method (str, optional): The filtering method used, can be 'Butter_bandpass', 'Butter_highpass',
                                          'Elliptic_bandpass', or 'Elliptic_highpass'. Default is 'Butter_bandpass'.
        order (int, optional): The order of the filtering. Default is 2.

    Returns:
        numpy.ndarray: The filtered signal. Shape = (recorded data points, number of electrodes).

    Raises:
        ValueError: If an invalid filtering method is chosen.
    """
    if filtering_method == "Butter_bandpass":
        lowcut = frequencies[0]
        highcut = frequencies[1]

        filtered = np.empty(shape=(raw_data.shape[0], raw_data.shape[1]))
        for i in range(raw_data.shape[1]):
            filtered[:, i] = butter_bandpass_filter(raw_data[:, i], lowcut, highcut, fsample=fsample, order=order)
        del recording_data

    elif filtering_method == "Butter_highpass":
        lowcut = frequencies[0]

        filtered = np.empty(shape=(raw_data.shape[0], raw_data.shape[1]))
        for i in range(raw_data.shape[1]):
            filtered[:, i] = butter_bandpass_filter_high(raw_data[:, i], lowcut, fsample=fsample, order=order)
        del recording_data

    elif filtering_method == "Elliptic_bandpass":
        lowcut = frequencies[0]
        highcut = frequencies[1]

        filtered = np.empty(shape=(raw_data.shape[0], raw_data.shape[1]))
        for i in range(raw_data.shape[1]):
            filtered[:, i] = ellip_filter(raw_data[:, i], lowcut, highcut, fsample=fsample, order=order)
        del recording_data

    elif filtering_method == "Elliptic_highpass":
        lowcut = frequencies[0]

        filtered = np.empty(shape=(raw_data.shape[0], raw_data.shape[1]))
        for i in range(raw_data.shape[1]):
            filtered[:, i] = ellip_filter_high(raw_data[:, i], lowcut, fsample=fsample, order=order)
        del recording_data

    else:
        raise ValueError("Please choose a valid filtering method! Chooose between Butter_bandpass, "
                         "Butter_highpass, Elliptic_bandpass or Elliptic_highpass")
    
    return filtered