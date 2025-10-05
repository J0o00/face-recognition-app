# Streamlit-Based Smart Recognition System
#
# Description:
# This application uses the Streamlit framework to create an interactive web app
# for face recognition. It allows a user to upload a photo and identifies any
# known individuals in that photo.

import face_recognition
import numpy as np
import os
import cv2
import streamlit as st
from PIL import Image

# --- Global Variables & Setup ---
KNOWN_FACES_FOLDER = 'known_faces'

# --- 1. Core Face Recognition Functions (from our previous app) ---

@st.cache_resource  # Decorator to cache the loaded faces
def load_known_faces():
    """Loads face encodings and names from the known_faces directory."""
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(KNOWN_FACES_FOLDER):
        os.makedirs(KNOWN_FACES_FOLDER)
    
    print("Loading known faces...")
    for filename in os.listdir(KNOWN_FACES_FOLDER):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            try:
                name = filename.rsplit('_', 1)[0]
                image_path = os.path.join(KNOWN_FACES_FOLDER, filename)
                person_image = face_recognition.load_image_file(image_path)
                
                encodings = face_recognition.face_encodings(person_image)
                if encodings:
                    # To handle multiple faces in one image, just add them all
                    for encoding in encodings:
                        known_face_encodings.append(encoding)
                        known_face_names.append(name)
                else:
                    print(f"Warning: No face found in {filename}. Skipping.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    print(f"Loaded {len(known_face_names)} known face encodings.")
    return known_face_encodings, known_face_names

def find_person_in_image(uploaded_image, known_face_encodings, known_face_names):
    """Detects and recognizes faces in an uploaded image and draws boxes."""
    # Convert the PIL image to an OpenCV image (NumPy array)
    frame = np.array(uploaded_image)
    # Convert RGB (from PIL) to BGR (for OpenCV)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Find faces
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_names = set()
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                recognized_names.add(name)
        
        # Draw a box around the face
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
    # Convert BGR (from OpenCV) back to RGB (for display)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), list(recognized_names)

# --- 2. Streamlit Web App Interface ---

# Load known faces once
known_encodings, known_names = load_known_faces()

st.set_page_config(page_title="Face Recognition App", layout="wide")
st.title("Smart Student Recognition System 👨‍🎓")
st.write("Upload a photo to identify registered students.")

# Note about the temporary file system
st.warning("ℹ️ **Note:** This is a demo app. The database of known faces is not persistent and will be reset if the app restarts.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Recognizing...")
    
    # Perform recognition
    result_image, recognized_names = find_person_in_image(image, known_encodings, known_names)
    
    # Display the result
    st.image(result_image, caption='Processed Image.', use_column_width=True)
    
    if recognized_names:
        st.success(f"**Found known student(s):** {', '.join(recognized_names)}")
    else:
        st.info("No known students were found in the image.")

st.sidebar.header("About")
st.sidebar.info("This app uses facial recognition to identify individuals. For this demo, there is no registration page. Known faces must be added manually to the GitHub repository.")