# Web-Based Smart Attendance System with Registration and Login
#
# Description:
# This is a multi-page Flask web application that provides a secure and robust
# face recognition system.
#
# Features:
# 1. Login System: A simple login page (hardcoded admin/password) to secure the app.
# 2. User Registration: A dedicated page for new users to register by providing their
#    name, class, and capturing three photos from their webcam.
# 3. Dual Recognition Modes:
#    - Live Recognition: Identifies users in real-time via the webcam.
#    - Upload Recognition: Identifies a user from an uploaded image.
# 4. Dynamic Face Loading: The system reloads known faces automatically after a new
#    user registers, without needing a server restart.

import face_recognition
import numpy as np
import csv
from datetime import datetime
import os
import base64
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import cv2
from functools import wraps

# --- 1. Flask App Initialization ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key'  # Change this in a real application!

# --- Global Variables & Setup ---
known_face_encodings = []
known_face_names = []
UPLOADS_FOLDER = 'uploads'
KNOWN_FACES_FOLDER = 'known_faces'

# --- 2. Helper Functions ---

def setup_directories():
    """Creates necessary directories if they don't exist."""
    for folder in [UPLOADS_FOLDER, KNOWN_FACES_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

def load_known_faces():
    """Loads face encodings and names from the known_faces directory."""
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()
    
    print("Loading known faces...")
    for filename in os.listdir(KNOWN_FACES_FOLDER):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            try:
                # The name is the part of the filename before the last underscore
                name = filename.rsplit('_', 1)[0] 
                image_path = os.path.join(KNOWN_FACES_FOLDER, filename)
                person_image = face_recognition.load_image_file(image_path)
                
                encodings = face_recognition.face_encodings(person_image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                else:
                    print(f"Warning: No face found in {filename}. Skipping.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    print(f"Loaded {len(known_face_names)} known faces.")

def find_person(frame):
    """Detects and recognizes faces in a given frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_names = set() # Use a set to avoid duplicate names in one frame
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        if name != "Unknown":
            recognized_names.add(name)
            
    return list(recognized_names)

# --- 3. Login & Session Management ---

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Hardcoded credentials for simplicity
        if request.form.get('username') == 'admin' and request.form.get('password') == 'password':
            session['logged_in'] = True
            flash('You were successfully logged in!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were logged out.', 'info')
    return redirect(url_for('login'))

# --- 4. Main Application Routes ---

@app.route('/')
@login_required
def index():
    """Serves the main recognition dashboard."""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    """Handles new user registration."""
    if request.method == 'POST':
        name = request.form.get('name')
        user_class = request.form.get('class')
        image_data = [request.form.get('image1'), request.form.get('image2'), request.form.get('image3')]
        
        if not all([name, user_class, all(image_data)]):
            flash('All fields and three photos are required!', 'danger')
            return redirect(url_for('register'))

        # Format: Name-Class
        user_identifier = f"{name}-{user_class}"
        
        for i, img_data_url in enumerate(image_data):
            try:
                header, encoded = img_data_url.split(",", 1)
                binary_data = base64.b64decode(encoded)
                filename = f"{user_identifier}_{i+1}.jpg"
                filepath = os.path.join(KNOWN_FACES_FOLDER, filename)
                with open(filepath, 'wb') as f:
                    f.write(binary_data)
            except Exception as e:
                flash(f'Error saving photo {i+1}. Please try again.', 'danger')
                print(f"Error saving photo: {e}")
                return redirect(url_for('register'))
        
        # Reload faces to include the new user
        load_known_faces()
        flash(f'User {name} registered successfully!', 'success')
        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/recognize_live', methods=['POST'])
@login_required
def recognize_live():
    """Endpoint for live webcam recognition."""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    header, encoded = data['image'].split(",", 1)
    binary_data = base64.b64decode(encoded)
    nparr = np.frombuffer(binary_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    recognized_names = find_person(frame)
    return jsonify({'names': recognized_names})

@app.route('/recognize_upload', methods=['POST'])
@login_required
def recognize_upload():
    """Endpoint for uploaded image recognition."""
    if 'photo' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['photo']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))

    if file:
        filepath = os.path.join(UPLOADS_FOLDER, file.filename)
        file.save(filepath)
        
        # Read the saved image for recognition
        uploaded_image = cv2.imread(filepath)
        recognized_names = find_person(uploaded_image)
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        if recognized_names:
            flash(f'Recognized student(s): {", ".join(recognized_names)}', 'success')
        else:
            flash('Could not recognize anyone in the uploaded photo.', 'warning')
            
    return redirect(url_for('index'))

# --- 5. Run the Flask App ---
if __name__ == '__main__':
    setup_directories()
    load_known_faces()
    app.run(debug=True, port=5000)

