#!/usr/bin/env python3.11
# Jesse Hitch - JesseBot@Linux.com
from flask import Flask, request, flash, redirect, send_file
import gzip as compress
import logging as log
from os import path
import sys
import utils
from werkzeug.utils import secure_filename


# set logging
log.basicConfig(stream=sys.stderr, level=log.INFO)
log.info("logging config loaded")

UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'}


app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def gzip_file(file_to_compress):
    """
    quickly gzip a file
    """
    fp = open(file_to_compress, "rb")
    data = fp.read()
    bindata = bytearray(data)

    with compress.open(f"{file_to_compress}.gz", "wb") as f:
        f.write(bindata)


@app.route('/infer_image/', defaults={'gzip': 0})
@app.route('/infer_image/<int:gzip>', methods=['GET', 'POST'])
def infer_image(gzip):
    """
    Runs utils.infer_image() on file upload
    Defaults to returning pkl type from ndarry.dump. if gzip != 0 return gzip
    """
    log.info("Accessed /infer_image")
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
            file_location = path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_location)
            log.info(f"Recieved file: {filename}")
            res = utils.infer_image(file_location)
            return_pkl = f"{filename}.pkl"
            res.dump(return_pkl)
            if gzip != 0:
                return_pkl = gzip_file(return_pkl)
            return send_file(return_pkl, as_attachment=True)

    # if they're not posting, show the upload page
    return 'upload a file to this endpoint'


if __name__ == '__main__':
    app.run(host='0.0.0.0')
