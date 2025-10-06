# Streamlit-Based Smart Recognition System using DeepFace
#
# Description:
# This application uses the modern DeepFace library and the Streamlit framework
# to create a reliable face recognition app.

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from deepface import DeepFace
import os
import cv2

# --- Setup and Configuration ---
st.set_page_config(page_title="Face Recognition", layout="wide")
KNOWN_FACES_DIR = "known_faces"

# Create the directory for known faces if it doesn't exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# --- Core Functions ---

def draw_annotations(image, results_df):
    """Draws bounding boxes and names on the image."""
    img_array = np.array(image)
    for index, row in results_df.iterrows():
        # DeepFace returns face coordinates in 'source_x', 'source_y', etc.
        x, y, w, h = row['source_x'], row['source_y'], row['source_w'], row['source_h']
        
        # Identity is the path to the matched image. We extract the name from it.
        identity_path = row['identity']
        # Extract the name from the file path (e.g., "known_faces/John_Doe.jpg" -> "John Doe")
        name = os.path.basename(identity_path).split('.')[0].replace('_', ' ')

        # Draw the bounding box
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Prepare text label
        label = f"{name}"
        
        # Calculate text size to draw a background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img_array, (x, y - text_height - 10), (x + text_width, y), (0, 255, 0), -1)
        cv2.putText(img_array, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
    return Image.fromarray(img_array)

# --- Streamlit App UI ---

st.title("👨‍🎓 Smart Student Recognition System")
st.write("This app uses the DeepFace library to identify students from a photo.")

st.info("To add a new student, place their photo in the `known_faces` folder in the GitHub repository.")

# Check if the known_faces directory is empty
if not os.listdir(KNOWN_FACES_DIR):
    st.warning("The 'known_faces' directory is empty. The app cannot recognize anyone until you add images to it on GitHub.")

uploaded_image = st.file_uploader("Upload an image to find registered students...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    # Convert image to numpy array for deepface
    img_np = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Finding matches... this might take a moment."):
        try:
            # The core of the new library: find matching faces
            # It searches the KNOWN_FACES_DIR for matches to faces in img_np
            dfs = DeepFace.find(
                img_path=img_np,
                db_path=KNOWN_FACES_DIR,
                enforce_detection=False, # Don't crash if a face isn't found
                silent=True # Suppress console output
            )
            
            # The result is a list of DataFrames, one for each face found in the uploaded image
            if dfs and not dfs[0].empty:
                result_df = dfs[0]
                processed_image = draw_annotations(image, result_df)
                
                with col2:
                    st.image(processed_image, caption="Recognition Results", use_column_width=True)
                
                # Extract and display names of recognized individuals
                identities = result_df['identity'].apply(lambda x: os.path.basename(x).split('.')[0].replace('_', ' '))
                st.success(f"**Recognized:** {', '.join(identities.unique())}")
            else:
                with col2:
                    st.image(image, caption="No matches found.", use_column_width=True)
                st.warning("Could not find any known students in the uploaded image.")

        except Exception as e:
            st.error(f"An error occurred during face recognition: {e}")

