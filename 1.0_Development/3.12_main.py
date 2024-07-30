import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from predictClassFromImage import unifyImageSize, predictClassProbabilityFromImage
from flask import Flask, render_template, request, jsonify, send_from_directory
from db_models import app, db, MedicalPrediction

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# LOAD THE TRAINED MODEL
model_path = ['2_savedModels/Br35H_model_97.67.h5', '2_savedModels/melanoma_cancer_model_90.99.h5', '2_savedModels/chest_xray_model_96.09.h5']
model = load_model(model_path[1])
print("Following model is loaded: ",model_path[1] )

# DEFINE VARIABLES
folder_path = 'static/1_doctorsImages/'
responseImageNames = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            print('No file part')
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            print('No selected file')
            return jsonify({'error': 'No selected file'})

        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            # Now you have the uploaded file saved in the 'uploads' folder

            filename = filename.replace("\\", "/")

            # IMAGE PROCESSING CODE GOES HERE

            # ADAPT IMAGE SHAPE
            unifyImageSize(filename, 60, 60, 2)

            # EXAMPLE USAGE:
            predicted_class, predicted_probability = predictClassProbabilityFromImage(model, filename)

            print(predicted_class, predicted_probability)
            print(responseImageNames)

            # Save prediction to the database
            new_prediction = MedicalPrediction(
                predicted_class=int(predicted_class),
                predicted_probability=str(predicted_probability)
            )
            db.session.add(new_prediction)
            db.session.commit()

            # Return the result as JSON
            result = {'predicted_class': int(predicted_class), 'predicted_probability': predicted_probability, "responseImageNames": responseImageNames}
            print(result)
            return jsonify({'result': result})

    except Exception as e:
        # Print the exception information for debugging
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': 'Internal server error'})

@app.route('/switchChange', methods=['POST'])
def switch_change_route():
    global model

    selected_switch_index = int(request.json.get('selectedSwitchIndex'))

    # Update the model_path based on the selected switch index
    selected_model_path = model_path[selected_switch_index - 1]
    model = load_model(selected_model_path)
    print("Following model is loaded: ", selected_model_path)

    # Perform any additional actions based on the selected switch
    # Return a response if needed
    return jsonify({'message': selected_model_path})

# Serve static files from the "static" directory
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
