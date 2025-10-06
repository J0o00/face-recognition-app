# Smart Student Recognition System using DeepFace
# This modern version uses the deepface library for robust, pre-trained face recognition.
# It is designed for easy deployment on cloud platforms like Streamlit Community Cloud.

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os
from deepface import DeepFace

# --- Configuration ---
DB_PATH = "known_faces" # Directory to store the database of known faces
MODEL_NAME = "VGG-Face" # Model for face recognition. Others: "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace"
DISTANCE_METRIC = "cosine" # Metric to measure similarity. Others: "euclidean", "euclidean_l2"

# --- Helper Functions ---

def setup_database():
    """Create the database directory if it doesn't exist."""
    if not os.path.exists(DB_PATH):
        st.info(f"Creating a directory for known faces at: {DB_PATH}")
        os.makedirs(DB_PATH)

@st.cache_data # Cache the database to avoid reloading on every interaction
def get_database_contents():
    """Lists the contents of the face database."""
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        return []
    # Return a list of student names from the filenames
    return sorted([f.split('.')[0] for f in os.listdir(DB_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))])

# --- Streamlit App UI ---

# Set up the main page
st.set_page_config(page_title="Student Recognition", layout="wide", initial_sidebar_state="expanded")
st.title("🎓 Smart Student Recognition System")
st.write("---")

# Initialize the database
setup_database()

# --- Sidebar for Registration ---
with st.sidebar:
    st.header("Register New Student")
    
    student_name = st.text_input("Enter Student Name:", placeholder="e.g., John Doe")
    new_photo = st.file_uploader("Upload Student Photo:", type=["jpg", "jpeg", "png"], help="Upload a clear, front-facing photo.")
    register_button = st.button("Register Student", use_container_width=True)

    if register_button and student_name and new_photo:
        with st.spinner("Processing registration..."):
            try:
                # Construct the file path
                file_path = os.path.join(DB_PATH, f"{student_name}.jpg")
                
                # Check if student already exists
                if os.path.exists(file_path):
                    st.warning(f"A student with the name '{student_name}' is already registered.")
                else:
                    # Save the uploaded photo
                    with open(file_path, "wb") as f:
                        f.write(new_photo.getbuffer())
                    
                    # Verify if a face can be detected in the new photo
                    try:
                        # The find function will raise an exception if no face is detected
                        DeepFace.find(img_path=file_path, db_path=DB_PATH, model_name=MODEL_NAME, distance_metric=DISTANCE_METRIC, enforce_detection=True)
                        st.success(f"✅ Successfully registered {student_name}!")
                        st.balloons()
                        # Clear cache to update the student list
                        st.cache_data.clear()
                    except ValueError as e:
                        # If DeepFace can't find a face, delete the file and inform the user
                        os.remove(file_path)
                        st.error("Registration failed. No face could be detected in the uploaded photo. Please use a clearer picture.")
            except Exception as e:
                st.error(f"An error occurred during registration: {e}")

    # Display the list of registered students
    st.write("---")
    st.header("Registered Students")
    registered_students = get_database_contents()
    if registered_students:
        st.dataframe(pd.DataFrame(registered_students, columns=["Name"]), use_container_width=True)
    else:
        st.info("No students registered yet. Use the form above to add one.")

# --- Main Area for Recognition ---
st.header("🔍 Recognize a Student")
recognition_photo = st.file_uploader("Upload a photo for recognition:", type=["jpg", "jpeg", "png"], key="recognition")

if recognition_photo is not None:
    # Check if there are any registered students
    if not get_database_contents():
        st.warning("Please register at least one student in the sidebar before attempting recognition.")
    else:
        with st.spinner("Finding matches..."):
            try:
                # Convert uploaded file to a format DeepFace can use
                image = Image.open(recognition_photo)
                img_array = np.array(image)

                # Use DeepFace to find the best match in the database
                # This function returns a list of pandas DataFrames. If empty, no match was found.
                dfs = DeepFace.find(
                    img_path=img_array,
                    db_path=DB_PATH,
                    model_name=MODEL_NAME,
                    distance_metric=DISTANCE_METRIC,
                    enforce_detection=False # Allow processing even if the main face is hard to detect
                )

                if dfs and not dfs[0].empty:
                    # Get the top match
                    best_match = dfs[0].iloc[0]
                    identity = best_match['identity']
                    
                    # Extract the name from the file path
                    recognized_name = os.path.basename(identity).split('.')[0]
                    
                    st.success(f"**Match Found!** This looks like **{recognized_name}**.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                    with col2:
                        st.image(identity, caption=f"Matched Image of {recognized_name}", use_column_width=True)
                else:
                    st.error("No match found in the database.")
                    st.image(image, caption="Uploaded Image", use_column_width=True)

            except ValueError as e:
                 st.error("Recognition failed. No face could be detected in the uploaded photo.")
            except Exception as e:
                st.error(f"An error occurred during recognition: {e}")

