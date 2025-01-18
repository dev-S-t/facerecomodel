# Server-side (Flask app - app.py)
import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import json
import time

# Load the trained model
print("Loading the model...")
model = tf.keras.models.load_model("emotion_model.h5")
print("Model loaded successfully.")

# Emotion mapping (from index to emotion name)
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Wellness suggestions for emotions
wellness_suggestions = {
    "happy": "Balanced state! Keep following your current routine.",
    "angry": "Pitta imbalance detected. Suggestions: cooling foods like cucumber, meditation, and coconut water.",
    "sad": "Vata imbalance detected. Suggestions: warm milk, sesame oil massage, and grounding exercises.",
    "neutral": "Maintain mindfulness to stay balanced.",
    "fear": "Practice calming techniques such as breathing exercises or guided meditation.",
    "disgust": "Engage in activities that bring joy and relaxation, such as art or nature walks.",
    "surprise": "Take time to process unexpected events mindfully and maintain emotional balance."
}

# Create Flask app
app = Flask(__name__)

# Route for emotion prediction (accepting preprocessed image data)
@app.route('/predict', methods=['POST'])
def predict_emotion():
    print("Received a request at /predict")
    try:
        # Get the preprocessed image data from the request
        print("Attempting to get JSON data from request...")
        data = request.get_json()
        print(f"Received JSON data: {data}")
        if not data or 'image_data' not in data:
            print("Error: No image_data found in request")
            return jsonify({"error": "No image_data found in request"}), 400

        # Convert the list back to a NumPy array
        print("Converting image_data to NumPy array...")
        image_data = np.array(data['image_data'])
        print(f"Image data shape after conversion: {image_data.shape}")

        # Ensure the image data has the correct shape
        if image_data.shape != (48, 48):
            print(f"Error: Incorrect image_data shape. Expected (48, 48), got {image_data.shape}")
            return jsonify({"error": f"Incorrect image_data shape. Expected (48, 48), got {image_data.shape}"}), 400

        # Expand dimensions to match the model's input shape (batch_size, height, width, channels)
        print("Expanding dimensions of image data...")
        img = np.expand_dims(image_data, axis=-1)  # Add channel dimension
        print(f"Image data shape after adding channel dimension: {img.shape}")
        img = np.expand_dims(img, axis=0)   # Add batch dimension
        print(f"Image data shape after adding batch dimension: {img.shape}")

        # Make prediction using the model
        print("Making prediction...")
        start_time = time.time()
        prediction = model.predict(img)
        end_time = time.time()
        print(f"Prediction time: {end_time - start_time:.4f} seconds")
        predicted_class = np.argmax(prediction)  # Get index of the highest probability
        predicted_emotion = emotion_labels.get(predicted_class, "Unknown")  # Map index to emotion
        print(f"Predicted class: {predicted_class}, Predicted emotion: {predicted_emotion}")

        # Get wellness suggestion
        wellness_advice = wellness_suggestions.get(predicted_emotion.lower(), "Stay mindful and balanced!")

        # Return prediction and advice
        print("Returning prediction and advice...")
        return jsonify({
            "emotion": predicted_emotion,
            "suggestion": wellness_advice
        })

    except ValueError as ve:
        print(f"ValueError occurred: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting the Flask app...")
    app.run(debug=True)
