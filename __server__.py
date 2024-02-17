# Test setting up a Flask server that handles images using PIL and CV2
import os

# server components
from flask import Flask
from flask import render_template
from flask import request, redirect, flash, url_for

# secure filename handling
from werkzeug.utils import secure_filename

# local external
import image_methods

# math
import numpy as np
import random
import base64

UPLOAD_FOLDER = "/images"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    """
    Front page for web application that ought to contain:
    * Upload image
    * Generate face recognition result
    * Draw new image
    * Display new image
    """
    image = request.args.get("img")
    return render_template("index.html?image="+image)

@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
    """
    1. Upload image.
    2. Store to disk.
    3. Send image to face detection.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part")
            return redirect(request.url)
        image = request.files['file']
        if image.filename == '':
            flash("No selected file")
            return redirect(request.url)
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            # Save to disk
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))
        img = Image.open(image)
        img = np.array(img)
    # Redirect with image file path to face detection.
    return redirect("/detect-faces/?img="+str(filename))
        
        
@app.route("/detect-faces")
def detect_faces():
    """
    1. Receive img parameter with URL of uploaded image.
    2. Do face recognition and generate a new file name.
    3. Upload the result back and relay image to display.
    """
    