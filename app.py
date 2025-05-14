import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model
from flask_cors import CORS
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # To allow frontend to connect (CORS = Cross Origin Resource Sharing)

# Load your trained model
model = load_model('paddyleaf_model.h5')

# Define the class names
paddyleaf_names = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot', 'not_paddy_leaf']

solutions = {
    'bacterial_leaf_blight': "Apply copper-based fungicides and remove infected leaves.",
    'brown_spot': "Use resistant varieties and practice crop rotation.",
    'healthy': "No action needed, the leaf is healthy.",
    'leaf_blast': "Use fungicides and practice proper irrigation.",
    'leaf_scald': "Reduce leaf wetness and use fungicides.",
    'narrow_brown_spot': "Apply fungicides and remove infected plant debris.",
    'not_paddy_leaf': "The uploaded image is not a paddy leaf. Please try again."
}

# Initialize the history list
history = []

# Function to convert image to base64 string
def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Prediction function
def classify_image(image):
    input_image = image.resize((180, 180))
    input_image_array = np.array(input_image)

    # Ensure image has 3 channels (RGB)
    if input_image_array.shape[-1] != 3:
        input_image_array = input_image_array[..., :3]

    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])

    predicted_class = paddyleaf_names[np.argmax(result)]
    prediction_score = np.max(result) * 100
    solution = solutions.get(predicted_class, "No solution available.")
    
    return predicted_class, prediction_score, solution

# API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400
    
    file = request.files['file']

    if file.filename == '':
        return "No file selected!", 400

    try:
        image = Image.open(file)
        predicted_class, prediction_score, solution = classify_image(image)

        # Convert image to base64
        image_base64 = convert_image_to_base64(image)

        # Save prediction to history with image
        history.insert(0,{
            'prediction': predicted_class,
            'confidence': f"{prediction_score:.2f}%",
            'solution': solution,
            'image': image_base64  # Store base64 encoded image
        })

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{prediction_score:.2f}%",
            'solution': solution,
            'image': image_base64  # Send back the image as base64
        })
    except Exception as e:
        return str(e), 500

# API route to get history
@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'history': history})

# Route to clear the history
@app.route('/clear_history', methods=['POST'])
def clear_history():
    global history
    history = []  # Clear the history list
    return jsonify({'message': 'History cleared successfully!'}), 200

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
