# model.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ===============================
# Load pretrained emotion model
# ===============================
model = load_model('emotion_model.h5')
print("✅ Emotion model loaded successfully.")

# Emotion classes (must match your model’s output order)
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ===============================
# Emotion-based recommendations
# ===============================
recommendations = {
    'Happy': "Keep smiling! Share your joy with others today.",
    'Sad': "It’s okay to feel down sometimes. Try taking a walk or calling a friend.",
    'Angry': "Take a deep breath. Calmness brings clarity.",
    'Fear': "Courage doesn’t mean no fear, but acting despite it.",
    'Disgust': "Step away from what’s bothering you and refocus your energy.",
    'Surprise': "Wow! Embrace the unexpected moments in life.",
    'Neutral': "A balanced mood is great for productivity."
}

# ===============================
# Emotion prediction function
# ===============================
def predict_emotion(img_path):
    """
    Load image, preprocess, and predict emotion.
    """
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]

    # Predict emotion
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds)
    emotion = classes[predicted_index]

    # Retrieve description
    description = recommendations.get(emotion, "Stay positive and mindful!")

    return emotion, description
