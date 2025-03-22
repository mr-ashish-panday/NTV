from flask import Flask, render_template, Response, jsonify
import cv2
import time
import torch
from torchvision.transforms import transforms
import pygame
import mediapipe as mp
import numpy as np
import os
import requests

# Adjusted paths: '../' moves up from api/ to the root directory
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Initialize pygame mixer for playing sounds
pygame.mixer.init()
# Initialize MediaPipe Hands model (commented out since mediapipe is disabled)
# mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the transformations for preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Download the model at runtime to reduce memory usage
model_path = "classical2.pth"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?export=download&id=1Qets_7ihMvX2bKfbpM459kQB57X6nUGd"
    print("Downloading model...")
    response = requests.get(url, stream=True)
    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model downloaded.")

# Load the PyTorch model
model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
model.eval()

# Define the class labels and corresponding Nepali labels
labels = ["आज", "धन्यबाद", "घर", "जान्छु", "म", "नमस्कार"]

# Corresponding .wav files adjusted to point to the static folder in the root
label_sounds = [
    "../static/audio/Aaja.wav",
    "../static/audio/Dhanyabaad.wav",
    "../static/audio/Ghar.wav",
    "../static/audio/Jaanchu.wav",
    "../static/audio/Ma.wav",
    "../static/audio/Namaskaar.wav"
]

gesture_delay = 4  # Delay in seconds between predictions
countdown_time = 4
last_prediction_time = 0
sentence = ""  # To store the current sentence
completed_sentences = []  # To store completed sentence
prediction_labels = ''  # To store predicted label
confidence = ''  # To store confidence percent
last_detected_gesture = None
current_prediction = {'label': '', 'confidence': '', 'sentence': '', 'completed_sentences': ''}
detection_enabled = False

def generate_frames():
    # Placeholder for Vercel - Load a static image from the root's static folder
    frame = cv2.imread("../static/placeholder.jpg")  # Ensure this file exists in static/
    if frame is None:
        # Fallback if image not found
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # Black 1280x720 image
        cv2.putText(frame, "No webcam on Vercel", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/start')
def start():
    global detection_enabled
    detection_enabled = True
    return jsonify(success=True)

@app.route('/stop')
def stop():
    global detection_enabled
    detection_enabled = False
    return jsonify(success=True)

@app.route('/clear')
def clear():
    global completed_sentences
    completed_sentences = ""  # Clear the sentence variable
    return jsonify(success=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/view')
def view():
    return render_template('view.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/letter')
def letter():
    return render_template('letters.html')

@app.route('/number')
def number():
    return render_template('number.html')

@app.route('/get_data')
def get_data():
    global sentence, completed_sentences, prediction_labels, confidence
    last_prediction = sentence.split(" ")[-1] if sentence else None
    return jsonify(current=sentence, completed=completed_sentences, prediction_label=prediction_labels, last_prediction=last_prediction, confidence=confidence)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use PORT from env, default to 8000
    app.run(debug=False, host="0.0.0.0", port=port)