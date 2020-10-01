import numpy as np


# Functions for preprocessing and preprocess function
def fft(iq, axis=0):
    """
    Computes the log of discrete Fourier Transform (DFT).
        
    Arguments:
    iq_burst -- {ndarray} -- 'iq_sweep_burst' array
    axis -- {int} -- axis to perform fft in (Default = 0)

    Returns:
    log of DFT on iq_burst array
    """
    iq = np.log(np.abs(np.fft.fft(hann(iq), axis=axis)))
    return iq


def hann(iq, window=None):
    """
    Preformes Hann smoothing of 'iq_sweep_burst'.

    Arguments:
      iq {ndarray} -- 'iq_sweep_burst' array
      window -{range} -- range of hann window indices (Default=None)
               if None the whole column is taken

    Returns:
      Regulazied iq in shape - (window[1] - window[0] - 2, iq.shape[1])
    """
    if window is None:
        window = [0, len(iq)]

    N = window[1] - window[0] - 1
    n = np.arange(window[0], window[1])
    n = n.reshape(len(n), 1)
    hannCol = 0.5 * (1 - np.cos(2 * np.pi * (n / N)))
    return (hannCol * iq[window[0]:window[1]])[1:-1]


def max_value_on_doppler(iq, doppler_burst):
    """
    Set max value on I/Q matrix using doppler burst vector. 
        
    Arguments:
    iq_burst -- {ndarray} -- 'iq_sweep_burst' array
    doppler_burst -- {ndarray} -- 'doppler_burst' array (center of mass)
                
    Returns:
    I/Q matrix with the max value instead of the original values
    The doppler burst marks the matrix values to change by max value
    """
    iq_max_value = np.max(iq)
    for i in range(iq.shape[1]):
        if doppler_burst[i]>=len(iq):
            continue
        iq[doppler_burst[i], i] = iq_max_value
    return iq


def normalize(iq):
    """
    Calculates normalized values for iq_sweep_burst matrix:
    (vlaue-mean)/std.
    """
    m = iq.mean()
    s = iq.std()
    return (iq-m)/s


def data_preprocess(data):
    """
    Preforms data preprocessing.
    Change target_type lables from string to integer:
    'human'  --> 1
    'animal' --> 0

    Arguments:
    data -- {ndarray} -- the data set

    Returns:
    processed data (max values by doppler burst, DFT, normalization)
    """
    X=[]
    for i in range(len(data['iq_sweep_burst'])):
        iq = fft(data['iq_sweep_burst'][i])
        iq = max_value_on_doppler(iq,data['doppler_burst'][i])
        iq = normalize(iq)
        X.append(iq)

    data['iq_sweep_burst'] = np.array(X)
    if 'target_type' in data:
        data['target_type'][data['target_type'] == 'animal'] = 0
        data['target_type'][data['target_type'] == 'human'] = 1
    return data