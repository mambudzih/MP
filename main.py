import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# TensorFlow Keras Model Prediction
def model_prediction(test_image):
    try:
        # Load the Keras model
        model = tf.keras.models.load_model("traine_model(1).keras")  # Update with your model path

        # Load and preprocess the image
        image = Image.open(test_image)  # Open the uploaded image
        image = image.resize((128, 128))  # Resize the image to the model's expected input size
        input_arr = np.array(image) / 255.0  # Normalize the image
        input_arr = np.expand_dims(input_arr, axis=0).astype(np.float32)  # Add batch dimension

        # Make predictions
        predictions = model.predict(input_arr)
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
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    ... (Your existing Home page text)
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ... (Your existing About page text)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        if st.button("Show Image"):
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
