# Client-side (example client.py)
import requests
from PIL import Image
import numpy as np
import json
import time

# Load and preprocess the image
image_path = "path/to/your/image.jpg"  # Replace with the actual path
print(f"Loading image from: {image_path}")
try:
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    print("Image loaded and converted to grayscale.")
except FileNotFoundError:
    print(f"Error: Image not found at {image_path}")
    exit()

print("Resizing image to 48x48...")
img = img.resize((48, 48))
print("Image resized.")

print("Converting image to NumPy array and normalizing...")
img_array = np.array(img).astype("float32") / 255.0
print("Image converted and normalized.")
print(f"Shape of preprocessed image array: {img_array.shape}")
print(f"Size of flattened array: {img_array.flatten().size}") # Added print statement

# Send the preprocessed image data to the Flask API
url = "YOUR_RENDER_APP_URL/predict"  # Replace with your Render app URL
print(f"Sending request to: {url}")
data = {'image_data': img_array.tolist()}  # Convert to list for JSON serialization
headers = {'Content-type': 'application/json'}
print(f"Sending data: {data}")

try:
    print("Sending the POST request...")
    start_time = time.time()
    response = requests.post(url, data=json.dumps(data), headers=headers)
    end_time = time.time()
    print(f"Request completed in: {end_time - start_time:.4f} seconds")

    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.text}")

    if response.status_code == 200:
        print("Prediction successful!")
        print(response.json())
    else:
        print("Prediction failed.")
        print(f"Error details: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred during the request: {e}")
