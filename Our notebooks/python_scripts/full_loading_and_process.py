from loading_functions import load_csv_metadata, load_data, append_dict
from sampling_data import subsampling, split_train_val, split_x_y
from data_preprocessing_funcs import data_preprocess
import numpy as np


# paths for the datasets
experiment_auxiliary_path = 'MAFAT RADAR Challenge - Auxiliary Experiment Set V2'
synthetic_auxiliary_path = 'MAFAT RADAR Challenge - Auxiliary Synthetic Set V2'
train_path = 'MAFAT RADAR Challenge - Training Set V1'
test_path = 'MAFAT RADAR Challenge - Public Test Set V1'


def process_and_split_data(training_df, synthetic_auxiliary_df, experiment_auxiliary_df, data_extraction):

  # load metadata of datasets
  exp_metadata = load_csv_metadata(experiment_auxiliary_path)
  synt_metadata = load_csv_metadata(synthetic_auxiliary_path)
  train_metadata = load_csv_metadata(train_path)

  print("exp_df", data_extraction['exp_df'][0], data_extraction['exp_df'][1])
  print("synt_df", data_extraction['synth_df'][0], data_extraction['synth_df'][1])
  print("train_df", data_extraction['train_df'][0], data_extraction['train_df'][1])
  print("valid_df", data_extraction['valid_df'][0], data_extraction['valid_df'][1])


  # Taking sample from the Auxiliary Experiment sets -Experiment + Synthetic
  exp_df = subsampling(experiment_auxiliary_df, exp_metadata, num_segments=data_extraction['exp_df'][0], balance_flag=data_extraction['exp_df'][1])
  synt_df = subsampling(synthetic_auxiliary_df,synt_metadata, num_segments=data_extraction['synth_df'][0], balance_flag=data_extraction['synth_df'][1])

  print("exp_df", f"humans segments: {(exp_df['target_type'] == 'human').sum()}", (exp_df['target_type'] == 'animal').sum())
  print("synt_df", f"humans segments: {(synt_df['target_type'] == 'human').sum()}", (synt_df['target_type'] == 'animal').sum())


  #-------------

  # spliting and sampling train and validation sets from the basic training data
  train_df, valid_df = split_train_val(training_df,
                                       train_metadata,
                                       train_seg_track=data_extraction['train_df'][0], valid_seg_track=data_extraction['valid_df'][0],
                                       balance_train=data_extraction['train_df'][1], balance_valid=data_extraction['valid_df'][1])
  
  print("train_df", f"humans segments: {(train_df['target_type'] == 'human').sum()}", (train_df['target_type'] == 'animal').sum())
  print("valid_df", f"humans segments: {(valid_df['target_type'] == 'human').sum()}", (valid_df['target_type'] == 'animal').sum())

   # Adding segments from the experiment auxiliary set to the training set
  print('[INFO] Adding segments from the experiment and Synthetic auxiliary sets to the training set')
  train_df = append_dict(train_df, exp_df)
  train_df = append_dict(train_df, synt_df)

  # Preprocessing and split the data to training and validation
  print('[INFO] Preprocessing and split the data to training and validation')
  train_df = data_preprocess(train_df)
  valid_df = data_preprocess(valid_df)
  train_x, train_y, val_x, val_y = split_x_y(train_df, valid_df)
  val_y = val_y.astype(int)
  train_y = train_y.astype(int)

  #-------------
  # Creating 3 channels for the train and validation set
  print('[INFO] Creating 3 channels for the train and validation set')
  train_x = np.repeat(train_x[...,np.newaxis], 3, -1)
  val_x = np.repeat(val_x[...,np.newaxis], 3, -1)

  #-------------

  # Public test set - loading and preprocessing
  print('[INFO] Loading and preprocessing public test set')
  test_df = load_data(test_path)
  test_df = data_preprocess(test_df.copy())
  test_x = test_df['iq_sweep_burst']
  test_x = np.repeat(test_x[...,np.newaxis], 3, -1)

  return train_x, val_x, test_x, train_y, val_y, test_df