# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import time
#from extract_video import extract
from perfecthuman import export_movie
from visualize1 import visualize

UPLOAD_FOLDER = './video/'
ALLOWED_EXTENSIONS = set(['mp4'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER#add path


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Routing
@app.route('/')
def index():
    title = "welcom"
    return render_template('index.html', title=title)


@app.route('/post', methods=['POST'])
def post():
#    if request.method == 'POST':
          # check if the post request has the file part
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #process_video()
        return redirect(url_for('processing',filename=filename))
    return redirect(url_for('index'))

@app.route('/processing/<filename>')
def processing(filename):
    title = "doumo"
    return render_template('processing.html', filename=filename,title=title)

@app.route('/processed/<filename>')
def processed(filename):
    title = "doumoar"
    print("Task started!")
    export_movie(filename)
    visualize(filename)
    print("Task is done!")
    return redirect(url_for('download',filename=filename))

@app.route('/download/<filename>')
def download(filename):
    remove_mp4 = filename.rsplit('.', 1)[0]
    print remove_mp4
    title="goodbye"
    return render_template('download.html', title=title, remove_mp4=remove_mp4)




@app.route('/download_file/<filename>')
def download_file(filename):
    return send_from_directory(directory=app.config['UPLOAD_FOLDER'], filename=filename)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
