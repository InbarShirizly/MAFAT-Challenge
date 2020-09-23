"""
flask api for server:
- upload spectorgran track as a pickle file
- predict for each segments using given model
- present results of prediction with spectrograms in a server, along with image of the full track
- history page to present previous spectrograms in the server
"""

from flask import Flask, render_template, url_for, flash, redirect, request
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from serve_the_model.python_scripts.utils import save_images_and_csv, generate_track_and_segments_data

# configuratiom constants
UPLOAD_FOLDER = r".\serve_the_model\uploaded_track_files"
SEGMENTS_IMAGES_FOLDER = r".\serve_the_model\static\segment_images"
ALLOWED_EXTENSIONS = ('pkl')

template_dir = os.path.abspath('./serve_the_model/templates')

color_map_path = ".\serve_the_model/data_train/cmap.npy"
cm_data = np.load(color_map_path)
color_map = LinearSegmentedColormap.from_list('parula', cm_data)

# flask app with configuration
app = Flask(__name__, template_folder=template_dir)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTS_FOLDER'] = SEGMENTS_IMAGES_FOLDER
app.config['SECRET_KEY'] = os.urandom(16)
app.config['target_dict'] = {0: "animal", 1: "human"}


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html", title="About")


@app.route("/prediction", methods=['POST'])
def prediction():
    """
    - check files in the post request and validate the files posted is pickle
    - saves images of spectrograms and csv of data
    - present predictions and data of each segment in the track in the page
    """
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part', "error")
        return redirect(url_for('home'))
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file', 'warning')
        return redirect(url_for('home'))
    if not file.filename.endswith(ALLOWED_EXTENSIONS):
        flash('File selected is not valid', 'warning')
        return redirect(url_for('home'))

    flash(f'file {file.filename} uploaded', 'success')
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), "rb") as f:
        track_dict = pickle.load(f)

    full_track_dict, df = save_images_and_csv(app, track_dict)
    df['predictions'] = df['predictions'].round(4)
    segment_data = list(df.reset_index().T.to_dict().values())

    return render_template("prediction.html", segment_data=segment_data, full_track_dict=full_track_dict)


@app.route("/history")
def history():
    """
    present history of predictions - list of tracks with data
    """
    files = os.listdir(app.config['SEGMENTS_FOLDER'])
    if len(files) <= 3:
        flash('There is no history yet', 'warning')
        return redirect(url_for('home'))

    range_list, segments_list, full_track_dict_list = generate_track_and_segments_data(app, files)

    return render_template("history.html", segments_list=segments_list,
                                           full_track_dict_list=full_track_dict_list,
                                           range_list=range_list,
                                            title="history")


if __name__ == '__main__':
    app.run(debug=False)