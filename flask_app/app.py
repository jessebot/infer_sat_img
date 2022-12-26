#!/usr/bin/env python3.11
# Jesse Hitch - JesseBot@Linux.com
from flask import Flask, request, flash, redirect, url_for
import logging as log
from os import path
import sys
from utils import infer_image
from werkzeug.utils import secure_filename


# set logging
log.basicConfig(stream=sys.stderr, level=log.INFO)
log.info("logging config loaded")

UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/utils/infer_image')
def infer_sat_image(image_path):
    """
    Infer the satellite image
    Returns numpyarray
    """
    return infer_image(image_path)
