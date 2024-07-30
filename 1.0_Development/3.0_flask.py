from flask import Flask, render_template, request, jsonify,send_from_directory
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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

        # Add your image processing code here
        # Example: Call a function to process the image and return the result
        result = process_image(filename)

        return jsonify({'result': result})

    return jsonify({'error': 'Unexpected error'})


# Serve static files from the "static" directory
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

def process_image(filename):
    # Add your image processing code here
    # Example: This function may return the result of processing
    return f'Image processed: {filename}'

if __name__ == '__main__':
    app.run(debug=True)
