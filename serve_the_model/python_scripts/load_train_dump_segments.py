import os
import pickle
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from data_preprocessing_funcs import data_preprocess


data_path = r'..\data_train'


train_path = 'MAFAT RADAR Challenge - Training Set V1'
pickle_folder_path = "../tracks_pickle_files/"

color_map_path = "../data_train/cmap.npy"
cm_data = np.load(color_map_path)
color_map = LinearSegmentedColormap.from_list('parula', cm_data)


def load_data(file_path):
    """
    Reads all data files (metadata and signal matrix data) as python dictionary,
    the pkl and csv files must have the same file name.

    Arguments:
      file_path -- {str} -- path to the iq_matrix file and metadata file

    Returns:
      Python dictionary
    """

    pkl = load_pkl_data(file_path)
    meta = load_csv_metadata(file_path)
    data_dictionary = {**meta, **pkl}

    for key in data_dictionary.keys():
        data_dictionary[key] = np.array(data_dictionary[key])

    return data_dictionary


def load_csv_metadata(file_path):
  """
  Reads csv as pandas DataFrame (only Metadata).

  Arguments:
    file_path -- {str} -- path to csv metadata file

  Returns:
    Pandas DataFarme
  """
  path = os.path.join(data_path, file_path + '.csv')
  with open(path, 'rb') as data:
    output = pd.read_csv(data)
  return output

def load_pkl_data(file_path):
  """
  Reads pickle file as a python dictionary (only Signal data).

  Arguments:
    file_path -- {str} -- path to pickle iq_matrix file

  Returns:
    Python dictionary
  """
  path = os.path.join(data_path, file_path + '.pkl')
  with open(path, 'rb') as data:
    output = pickle.load(data)
  return output


def create_list_of_track_dicts(full_dict, train_path, min_segments=5, max_segments=12, num_tracks=30):
    df = load_csv_metadata(train_path)

    track_with_many_segments = (df.groupby(by="track_id")['segment_id'].count() > min_segments) & \
                               (df.groupby(by="track_id")['segment_id'].count() < max_segments)

    chosen_tracks = np.random.choice(track_with_many_segments[track_with_many_segments].index, size=num_tracks)

    tracks_data_list = []

    for track in chosen_tracks:
        filt = (df['track_id'] == track)
        track_segments = df.loc[filt, 'segment_id']

        new_dict = {}

        for key in ['segment_id', 'snr_type', "doppler_burst", "iq_sweep_burst", "target_type"]:
            new_dict[key] = full_dict[key][track_segments]

        new_dict['track_id'] = track

        data_preprocess(new_dict)
        a = new_dict['iq_sweep_burst']
        new_dict['full_track_iq'] = np.concatenate([a[i, :, :] for i in range(a.shape[0])], axis=1)

        tracks_data_list.append(new_dict)

    return tracks_data_list


def dump_tracks_to_pickles(tracks_data_list, pickle_folder_path):
    for i in range(len(tracks_data_list)):
        with open(pickle_folder_path + f"track_{tracks_data_list[i]['track_id']}.pkl", "wb") as f:
            pickle.dump(tracks_data_list[i], f)


if __name__ == '__main__':
    full_dict = load_data(train_path)
    tracks_data_list = create_list_of_track_dicts(full_dict, train_path, min_segments=5, max_segments=12, num_tracks=30)
    dump_tracks_to_pickles(tracks_data_list, pickle_folder_path)
