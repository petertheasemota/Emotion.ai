from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import os
import uuid

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained emotion recognition model
MODEL_PATH = 'emotion_model.h5'
model = load_model(MODEL_PATH)

# Define emotion classes (ensure this matches your training order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Optional: Emotion descriptions
emotion_descriptions = {
    'Angry': 'You seem upset or frustrated.',
    'Disgust': 'That face shows dislike or displeasure.',
    'Fear': 'Looks like you are frightened or anxious.',
    'Happy': 'A bright smile! You look happy.',
    'Neutral': 'A calm and neutral expression.',
    'Sad': 'You look sad or disappointed.',
    'Surprise': 'You look surprised or shocked.'
}

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get file from request (from webcam capture or upload)
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Save file temporarily
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Open and preprocess image
        image = Image.open(filepath).convert('RGB')
        image = image.resize((224, 224))  # MobileNetV2 input size
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        # Make prediction
        preds = model.predict(image_array)
        emotion_idx = np.argmax(preds[0])
        emotion = emotion_labels[emotion_idx]
        description = emotion_descriptions.get(emotion, "")

        # Return result
        return jsonify({'emotion': emotion, 'description': description})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True)
