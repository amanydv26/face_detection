from flask import Flask, render_template, request, redirect, url_for, jsonify
import tensorflow as tf
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = './backend/uploads'
MODEL_PATH = './backend/model/deepfake_detector.h5'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Preprocess the input (adjust as needed for your model)
    img = tf.keras.utils.load_img(filepath, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict using the model
    prediction = model.predict(img_array)
    result = "DeepFake" if prediction[0][0] > 0.5 else "Authentic"
    
    return jsonify({'result': result, 'confidence': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
