# Python script for streamlit
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import os
import streamlit as st
import joblib
from PIL import Image
import pandas as pd
import time

# Load the model 
model = tf.keras.models.load_model("streamlit_app/FINAL_bird_classifier_2.h5")

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
class_dictionary = pd.read_csv('streamlit_app/class_dictionary.csv', index_col=0)

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
    st.write("Do you have a picture of a bird that you can't identify? Our model can correctly identify the names of 525 different species of birds with roughly 98% accuracy.")

    # Insert a line or divider
    st.markdown("---")
    
    # Test out some of our images
    st.subheader("Give it a try!")
    
    # Load your Bird images (replace these paths with your actual file paths)
    image_paths = ["streamlit_app/Ant_Bird_1.jpg", "streamlit_app/Lazuli_Bunting_1.jpg", "streamlit_app/Peregrine_Falcon_1.jpg", "streamlit_app/Tawny_Frogmouth_1.jpg"]

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
    
    # Scrolling List of Viable Species
    st.write("Here's a list of birds in our training data:")
    
    formatted = pd.DataFrame(class_dictionary)
    formatted.columns = ["Species"]
    st.dataframe(formatted, width=300, hide_index=True)
    
    
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
    
    if 'no_clicked' not in st.session_state:
        st.session_state['no_clicked'] = False
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""

    st.write("Was our prediction correct?")

    if st.button("Yes"):
        # Reset the state when "Yes" is clicked
        st.session_state['no_clicked'] = False
        st.write("Our model is good at identifying those!")

    if st.button("No"):
        # Set a flag to keep track of the "No" button being clicked
        st.session_state['no_clicked'] = True

    # Check the session state to decide whether to show the input box
    if st.session_state['no_clicked']:
        # Use session_state to capture and persist user input
        st.session_state['user_input'] = st.text_input("What is the name of the bird you provided?")
        if st.session_state['user_input']:
            # Assuming 'classes' is a dictionary available in your app
            if st.session_state['user_input'].upper() in classes.values():
                st.write("The model should've gotten that correct, but we made an incorrect prediction")
            else:
                st.write("This type of bird was not present in our training data, sorry about that.")
    

    github_project_url = "https://github.com/dvdhartsman/Bird_Species_Image_Classification"
    github_project_markdown = f'[our GitHub repository]({github_project_url})'

    st.write("   ")
    st.markdown("---")
    st.write(f"This model is a convolutional neural network (CNN) image classification model created with Python, Tensorflow, and Keras. It is informed by the EfficientNetB0 architecture. The model currently has an approximate 98% accuracy score. Check out our work at {github_project_markdown}. Please feel free to connect with us on LinkedIn or via email.") 

# Sidebar - Bio info
st.sidebar.title('About Us:')

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
                         
# Heath display
heath_linkedin_url = "https://www.linkedin.com/in/heefjones/"
heath_github_url = "https://github.com/heefjones"
heath_medium_url = "https://medium.com/@heefjones"
                         
heath_linkedin_markdown = f'[LinkedIn]({heath_linkedin_url})'
heath_github_markdown = f'[GitHub]({heath_github_url})'
heath_medium_markdown = f'[Blog]({heath_medium_url})'                         
                         
st.sidebar.subheader('Heath Jones')
st.sidebar.markdown(f"{heath_linkedin_markdown} | {heath_github_markdown} | {heath_medium_markdown}", unsafe_allow_html=True)
st.sidebar.write('heefjones9@gmail.com')

# Run the app
if __name__ == "__main__":
    main()
