import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from predictClassFromImage import unifyImageSize,predictClassProbabilityFromImage
from flask import Flask, render_template, request, jsonify,send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# LOAD THE TRAINED MODEL
model_path = '2_savedModels/brain_tumor_model.h5'
model = load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        # Now you have the uploaded file saved in the 'uploads' folder

        filename=filename.replace("\\", "/")

        # IMAGE PROCESSING CODE GOES HERE

        # ADAPT IMAGE SHAPE
        unifyImageSize(filename, 60, 60, 2)

        # EXAMPLE USAGE:
        predicted_class, predicted_probability = predictClassProbabilityFromImage(model, filename)

        print(predicted_class,predicted_probability)

        return jsonify({'result': [predicted_class, predicted_probability]})

    return jsonify({'error': 'Unexpected error'})




# Serve static files from the "static" directory
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(debug=True)