import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Tensorflow Lite Model Prediction
def model_prediction(test_image):
    try:
        # Load the TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path="model(1).tflite")
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Load and preprocess the image
        image = Image.open(test_image)  # Open the uploaded image
        image = image.resize((128, 128))  # Resize the image
        input_arr = np.array(image) / 255.0  # Normalize the image
        input_arr = np.expand_dims(input_arr, axis=0).astype(np.float32)  # Add batch dimension

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_arr)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        predictions = interpreter.get_tensor(output_details[0]['index'])
        return np.argmax(predictions)  # Return index of max element

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Streamlit UI and logic remain the same
# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    # (Your existing Home page code)

# About Project
elif app_mode == "About":
    # (Your existing About page code)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        if st.button("Show Image"):
            # Display the uploaded image
            st.image(test_image, use_column_width=True)

        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            if result_index is not None:
                # Reading Labels
                class_name = [
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight',
                    'Corn_(maize)___healthy'
                ]
                st.success(f"Model predicts it's a {class_name[result_index]}")
