# Python script for streamlit
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model
import os
import streamlit as st
import joblib
from PIL import Image
import pandas as pd

# Load the model 
model = tf.keras.models.load_model("FINAL_bird_classifier_2.h5")

# For any given bird image, it needs to conform to an expected format (224,224,3)
def format_image(img_path):
    try:
        # Read the image in as numeric data
        img = mpimg.imread(img_path)
        # Resize the data to conform with the expected format
        resized_image = tf.image.resize(img, (224,224))
        # Return the image
        return resized_image
    except Exception as e:
        # Streamlit warning about non-conforming data
        st.warning("Please upload a valid image.")
        return None

# Dictionary of Class/Species Values 
# Load the csv, easier than manually writing the labels
class_dictionary = pd.read_csv('class_dictionary.csv', index_col=0)
# Create a dictionary with the correct labels
classes = class_dictionary.to_dict()["0"]

# Function gor generating actual predictions
def get_prediction(resized_image):
    # Handle the expectation of batch-size
    y_pred = model.predict(np.expand_dims(resized_image, 0))
    # Find the maximum value of the 525 class predictions
    pred_class = np.argmax(y_pred[0])
    # Access the predicted bird name
    class_name = classes[pred_class]
    return class_name.title()
    
    
def main():
    # Title the app
    st.title("Bird Species Classification")

    # Introduction to the purpose of the app
    st.write("Do you have a picture of a bird that you can't identify? This model can correctly identify the name of 525 different species of birds! This model is capable of correctly identifying birds with nearly 98% accuracy.")

    # Insert a line or divider
    st.markdown("---")
    
    # Test out some of our images
    st.subheader("Give it a try!")
    
    # Load your chest X-ray images (replace these paths with your actual file paths)
    image_paths = ["Ant_Bird_1.jpg", "Lazuli_Bunting_1.jpg", "Peregrine_Falcon_1.jpg", "Tawny_Frogmouth_1.jpg"]

    # Function to resize the image to a specified width and height -> for display purposes only
    def resize_image(image_path, width, height):
        image = Image.open(image_path)
        resized_image = image.resize((width, height))
        return resized_image
    
    # Specify the width and height for resizing
    image_width = 400
    image_height = 400
    
    # Use st.columns to create columns corresponding to the number of images we have uploaded for trial
    columns = st.columns(len(image_paths))
    
    # Mapping of image indices to results
    results_mapping = {
        1: "Antbird",
        2: "Lazuli Bunting",
        3: "Peregrine Falcon",
        4: "Tawny Frogmouth"
    }
    
    # Display each image in a column with a "Detect" button underneath
    for idx, (column, image_path) in enumerate(zip(columns, image_paths)):
        # Resize the image
        resized_image = resize_image(image_path, image_width, image_height)
        
        # Display the resized image with the specified width and center it
        column.image(resized_image, caption=f"Bird Image {idx + 1}", use_column_width=True)
        
        # Add a "Detect" button centered below the image
        if column.button(f"Identify Bird {idx + 1}"):
            # Display the spinner animation while processing
            with st.spinner(f"Results are loading for image {idx + 1}..."):
                # Simulate model processing time (replace with your actual detection logic)
                time.sleep(1)
        
                # Your detection logic goes here
                result = results_mapping.get(idx + 1, "Prediction Error")  
        
                # Display the result
                column.write(f"{result}")
    
    # Insert spacing with divider 
    st.markdown("---")
    
    st.subheader('Try your own bird image')
    st.write('Submit a bird image to identify which of the 525 known species it is!')

    # File uploader for image
    uploaded_file = st.file_uploader("Please select an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.")
    
        # Preprocess the image
        input_data = format_image(uploaded_file)
    
        # Add a "Detect" button
        if st.button("Predict Species"):
            
            # Display the spinner while processing
            with st.spinner("Identifying Species..."):
                # Simulate model processing time (replace with your actual detection logic)
                time.sleep(2)
                
                # Make predictions
                predictions = get_prediction(input_data)
    
                # Check if predictions is not None
                if predictions is not None:
                    # Display the predictions
                    st.write("### Results:")
                    st.write(predictions)
                else:
                    st.write("Unable to arrive at a prediction")

    st.write("   ")
    st.markdown("---")
    
    st.write("Were we correct?")
    if st.button("Yes"):
        st.write("Yeah, we thought so...")
    if st.button("No"):
        user_input = st.text_input("What is the name of the bird you provided? ")
        if user_input.upper in classes.values():
            st.write("The model should've known that, but we made an incorrect prediction")
        else: 
            st.write("This type of bird is not present in our training data, sorry about that")
    

    github_project_url = "https://github.com/dvdhartsman/Bird_Species_Image_Classification"
    github_project_markdown = f'[GitHub]({github_project_url})'

    st.write("   ")
    st.write(f"This model is based on a convolutional neural network (CNN) image classification model using Python, Tensorflow, and Keras, and it is informed by the EfficientNetB0 architecture. The model currently has an approximate 98% accuracy rate and can be found in {github_project_markdown}. Please feel free to connect with me on LinkedIn or via email.") 

# Sidebar - Bio info
st.sidebar.title('About Me:')

# Headshot maybe
# st.sidebar.image("app_images/headshot.jpg", use_column_width=True)

# Variables for f-strings
linkedin_url = "https://www.linkedin.com/in/david-hartsman-data/"
github_url = "https://github.com/dvdhartsman"
medium_url = "https://medium.com/@dvdhartsman"

linkedin_markdown = f'[LinkedIn]({linkedin_url})'
github_markdown = f'[GitHub]({github_url})'
medium_markdown = f'[Blog]({medium_url})'

# Text display
st.sidebar.subheader('David Hartsman')
st.sidebar.markdown(f"{linkedin_markdown} | {github_markdown} | {medium_markdown}", unsafe_allow_html=True)
st.sidebar.write('dvdhartsman@gmail.com')

# Run the app
if __name__ == "__main__":
    main()
