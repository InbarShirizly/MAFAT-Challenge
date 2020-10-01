import numpy as np


def recenter_midline(radar_array):
  """
  This function shifts the spectogram by 63 (half) upwards so that the center of
  the spectogram matches the center position. Anything above means going towards
  and values bellow mean the object is going away.
  """
  center = int(radar_array.shape[0]/2)
  center_radar = np.concatenate((radar_array[center:],radar_array[:center]))
  return center_radar


def shift_spectrogram(iq_array, shift=16):
    """
    The funtion return the spectoram shifted by the given value 
    the max value = 31 - must be only for one segment
    """
    return np.concatenate((iq_array[:, shift:], iq_array[:, :shift]), axis=1)