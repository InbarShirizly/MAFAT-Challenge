import numpy as np


# Function for splitting the data to training and validation
# and function for selecting samples of segments from the Auxiliary dataset
def split_train_val(data, meta_data_df, train_seg_track=1, valid_seg_track=1, balance_train=False, balance_valid=True):
    """
    spliting according to geo_id (1,4 for validation) and using subsampling function
    to each one of the functions. making sure as well to not pick segments from
    the synthetic dataset that have geolocation 1,4 but are not relevant to the validation
    """
    idx = ((data['geolocation_id'] == 4) | (data['geolocation_id'] == 1))

    train_df, valid_df = {}, {}
    for key in data:
        train_df[key] = data[key][np.logical_not(idx)]
        valid_df[key] = data[key][idx]


    train_metadata = meta_data_df.iloc[train_df['segment_id']]
    valid_metadata = meta_data_df.iloc[valid_df['segment_id']]

    train_df = subsampling(data, train_metadata, train_seg_track, balance_train)
    valid_df = subsampling(data, valid_metadata, valid_seg_track, balance_valid)

    return train_df, valid_df


def subsampling(data, meta_data_df, num_segments=3, balance_flag=True):

  """ sample the input dict, using function of subsampling on the meta data dataframe to
      find the relevant segemtns """

  segements_indices = subsampling_segments_target_ratio(meta_data_df, num_segments=num_segments, balance_flag=balance_flag)  
  new_df = {}

  for key in data:
      new_df[key] = data[key][segements_indices]

  return new_df

def subsampling_segments_target_ratio(df, num_segments, balance_flag):

  """ pick num_segments of each track - randomly shuffeled
      if balance flag is up - will try to undersample one of the target to balance
      the amount of segments of the animal or human. in a case that it's impossible - will
      print it and return the values as is
      input:  dataframe of the metadata 
      output: indexes needed"""

  segments_numbers = np.array([])

  for track in np.unique(df['track_id']):
      filt = (df['track_id'] == track)
      track_segments = df.loc[filt, 'segment_id'] 

      if len(track_segments) > num_segments:
        segments = np.random.choice(track_segments, num_segments, replace=False)
      else: 
        segments = track_segments
      segments_numbers = np.concatenate((segments_numbers, segments))
  
  filt = df['segment_id'].isin(segments_numbers.astype(int))
  relevant_indices = df[filt].index

  if balance_flag:
      relevant_df = df.loc[relevant_indices]
      relevant_indices = balance_target(relevant_df)

  return relevant_indices


def balance_target(relevant_df):

    human_segments = (relevant_df['target_type'] == 'human').sum()
    animal_segments = (relevant_df['target_type'] == 'animal').sum()
    min_segments = min(human_segments, animal_segments)

    if (human_segments != 0) and (animal_segments != 0):
      animal_indices = relevant_df[relevant_df['target_type'] == 'animal'].sample(n=min_segments).index
      human_indices = relevant_df[relevant_df['target_type'] == 'human'].sample(n=min_segments).index
      relevant_indices = np.concatenate((human_indices.to_numpy(), animal_indices.to_numpy()))

    elif (human_segments == 0) and (animal_segments == 0):
      print("This dataset will not be part of the data exrtaction")
      relevant_indices = relevant_df.index
    else:
      print(f"Data contains only humans or animals - can't balance")
      relevant_indices = relevant_df.index
    
    return relevant_indices

def split_x_y(train_df, valid_df):

    X_train = train_df['iq_sweep_burst']
    y_train = train_df['target_type']
    X_valid = valid_df['iq_sweep_burst']
    y_valid = valid_df['target_type']

    return X_train, y_train, X_valid, y_valid

