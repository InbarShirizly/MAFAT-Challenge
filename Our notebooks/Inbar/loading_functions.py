import os
import pickle
import pandas as pd
import numpy as np

mount_path = '/content/gdrive/'
competition_path = 'My Drive/Final Project ITC/MAFAT Challenge/Data'
experiment_auxiliary_path = 'MAFAT RADAR Challenge - Auxiliary Experiment Set V2'
synthetic_auxiliary_path = 'MAFAT RADAR Challenge - Auxiliary Synthetic Set V2'
train_path = 'MAFAT RADAR Challenge - Training Set V1'

# Functions for loading the data
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


def load_pkl_data(file_path):
  """
  Reads pickle file as a python dictionary (only Signal data).

  Arguments:
    file_path -- {str} -- path to pickle iq_matrix file

  Returns:
    Python dictionary
  """
  path = os.path.join(mount_path, competition_path, file_path + '.pkl')
  with open(path, 'rb') as data:
    output = pickle.load(data)
  return output


def load_csv_metadata(file_path):
  """
  Reads csv as pandas DataFrame (only Metadata).

  Arguments:
    file_path -- {str} -- path to csv metadata file

  Returns:
    Pandas DataFarme
  """
  path = os.path.join(mount_path, competition_path, file_path + '.csv')
  with open(path, 'rb') as data:
    output = pd.read_csv(data)
  return output


def load_data_all_datasets():

  print('[INFO] Loading and spliting the data')

  print('[INFO] Loading Auxiliary Experiment set - can take a few minutes')
  experiment_auxiliary_df = load_data(experiment_auxiliary_path)

  print('[INFO] Loading Auxiliary Synthetic set - can take a few minutes')
  synthetic_auxiliary_df = load_data(synthetic_auxiliary_path)

  print('[INFO] Loading Train set - can take a few minutes')
  training_df = load_data(train_path)

  return training_df, synthetic_auxiliary_df, experiment_auxiliary_df


# The function append_dict is for concatenating the training set 
# with the Auxiliary data set segments

def append_dict(dict1, dict2):
  new_dict = {}
  for key in dict1:
    new_dict[key] = np.concatenate([dict1[key], dict2[key]], axis=0)
  return new_dict