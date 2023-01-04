#!/usr/bin/env python3.11
# Jesse Hitch - JesseBot@Linux.com
from flask import Flask, request, redirect, send_file, url_for
# import gzip as compress
from os import path, environ
from werkzeug.utils import secure_filename
from app_logger import log
# utils should use logger above now
import utils


UPLOAD_FOLDER = '/tmp/'
GPU = environ.get('GPU', False)


app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'8754hfpOdF329rZhn'


def allowed_file(filename):
    """
    make sure this is an expected file extenion, tiff or tif
    """
    allowed_extensions = {'tif', 'tiff'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/infer_image/', methods=['GET', 'POST'])
def infer_image():
    """
    Runs utils.infer_image() on file upload
    Defaults to returning pickle from ndarry.dump
    """
    log.info("Accessed /infer_image/")
    if request.method == 'POST':
        log.info(request)
        # check if the post request has the file part
        if 'file' not in request.files:
            log.info('No file part')
            return redirect(request.url)
        file = request.files['file']

        # if no file selected
        if file.filename == '':
            log.info('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # make sure this is not insecure
            filename = secure_filename(file.filename)
            log.info(f"Recieved file: {filename}")

            # save the file locally
            file_location = path.join(UPLOAD_FOLDER, filename)
            file.save(file_location)
            log.info(f"Saved file: {filename}")

            # run the infer_image function for the assignment
            res = utils.infer_image(file_location, plot=False, use_gpu=GPU)

            # dump the numpy array to a pkl
            return_pkl = filename.replace('tif', 'pkl')
            log.info(f"Creating pickle for numpy array in: {return_pkl}")
            res.dump(return_pkl)

            # return file to user
            log.info(f"Returning pickle over HTTP: {return_pkl}")
            return send_file(return_pkl, as_attachment=True)

    # if they're not posting, show the upload page
    return 'upload a file to this endpoint'
