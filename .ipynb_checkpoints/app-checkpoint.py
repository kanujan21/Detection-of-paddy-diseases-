import os
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model
import streamlit as st

# Set up the Streamlit page title and header
st.title('🌾 Paddy Leaf Disease Identification')
st.header('Upload a paddy leaf image to identify the disease using a CNN model')

# Define the class names for the paddy leaf diseases
paddyleaf_names = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot']

# Load the trained model
model = load_model('paddyleaf_model.h5')

# Define a function to classify the image
def classify_image(image_path):
    """
    Classify an image into one of the classes from paddyleaf_names using the loaded model.
    """
    input_image = Image.open(image_path).resize((180, 180))
    input_image_array = np.array(input_image)

    # Ensure the image has 3 channels (RGB)
    if input_image_array.shape[-1] != 3:
        input_image_array = input_image_array[..., :3]

    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])

    predicted_class = paddyleaf_names[np.argmax(result)]
    prediction_score = np.max(result) * 100

    return f"The image belongs to **'{predicted_class}'** with a confidence of **{prediction_score:.2f}%**"

# Streamlit file uploader for the image
uploaded_file = st.file_uploader("Upload an Image of a Paddy Leaf", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Ensure 'uploads' folder exists
    os.makedirs('uploads', exist_ok=True)

    # Save the uploaded image to the 'uploads' folder
    file_path = os.path.join('uploads', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Display the prediction result
    prediction = classify_image(file_path)
    st.markdown(prediction)
