"""
helper functions for the flask api
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model


MODEL_PATH = r".\serve_the_model\transfer_learning_model.h5"
model = load_model(MODEL_PATH)


def save_images_and_csv(app, track_dict):
    """
    - generate a dataframe of data of segments from the given track_dict. including prediction from the model
    - save each of the segments images in a folder and the full track spectrogram
    :param: app - flask app from main file
    :param: track_dict - values of the given track (dict)
    :retrun full_track_dict - dict of data for the full_track,
            df - dataframe of data of segments from the given track_dict
    """
    track_num = track_dict['track_id']
    target_type = track_dict['target_type'][0]
    segment_image_names = []
    for i, segment in enumerate(track_dict['iq_sweep_burst']):
        segment_image_names.append(f"track_{track_num}_{i}.png")
        image_path = os.path.join(app.config['SEGMENTS_FOLDER'], segment_image_names[-1])
        plt.imsave(image_path, segment)

    full_track_image_path = os.path.join(app.config['SEGMENTS_FOLDER'], f"track_{track_num}_full_track.png")
    plt.imsave(full_track_image_path, track_dict['full_track_iq'])

    x_test = np.repeat(track_dict['iq_sweep_burst'][..., np.newaxis], 3, -1)
    segments_predictions = [pred[0] for pred in model.predict(x_test)]

    df = pd.DataFrame(
        list(zip(segment_image_names,
                 track_dict['segment_id'],
                 track_dict['snr_type'],
                 track_dict['target_type'],
                 segments_predictions)),
        columns=["image_name", "segment_id", "snr_type", "target", "predictions"])

    df.sort_values(by=['segment_id'], inplace=True)

    df.to_csv(os.path.join(app.config['SEGMENTS_FOLDER'], f'track_{track_num}.csv'))

    full_track_dict = {"name": f"track_{track_num}_full_track.png",
                       "target": app.config['target_dict'][target_type] + " == " + str(target_type),
                       "track_num": track_num}

    return full_track_dict, df


def generate_track_and_segments_data(app, files):
    """
    generate list of tracks data and segments for history presentation
    :param: app - flask app from main file
    :param: files - names of files in the directory where the spectrgrams located (list of str)
    :retrun range_list - list of numbers for html.
            segments_list - list of values relevant for segments in each track in the list
            full_track_dict_list - list of dictionaries of data for the full_track
    """
    tracks_df_dict = {}
    for file in files:
        if file.endswith(".csv"):
            tracks_df_dict[file.split(".")[0]] = pd.read_csv(os.path.join(app.config['SEGMENTS_FOLDER'],file))

    tracks_names = sorted(tracks_df_dict.keys(), key=lambda string: int(string.split("_")[-1]))
    range_list = range(len(tracks_names))
    segments_list, full_track_dict_list = [], []

    for i in range_list:
        df = tracks_df_dict[tracks_names[i]]
        df['predictions'] = df['predictions'].round(4)
        segments_list.append(list(df.T.to_dict().values()))
        full_track_dict_list.append({"name": tracks_names[i] + "_full_track.png",
                                   "target": app.config['target_dict'][df['target'][0]] + " == " + str(df['target'][i]),
                                   "track_num": tracks_names[i].split("_")[1]})

    return range_list, segments_list, full_track_dict_list