import os
import pickle
import torch
import requests
import io
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- 1) Initialize Img2Vec with GPU support if available ---
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("CUDA is available. Using GPU for embeddings.")
else:
    print("CUDA not available. Falling back to CPU for embeddings.")

img2vec = Img2Vec(cuda=use_cuda)

# --- 2) Load the saved scaler and classification model ---
with open('./scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('./model.pkl', 'rb') as f:
    classifier = pickle.load(f)

# --- 3) Define the mapping from numeric indices to traffic sign names ---
class_names = [
    "SPEED LIMIT 20",
    "SPEED LIMIT 30",
    "SPEED LIMIT 50",
    "SPEED LIMIT 60",
    "SPEED LIMIT 70",
    "SPEED LIMIT 80",
    "END OF SPEED LIMIT 80",
    "SPEED LIMIT 100",
    "SPEED LIMIT 120",
    "NO OVERTAKING",
    "NO OVERTAKING BY TRUCKS",
    "INTERSECTION WARNING",
    "PRIORITY ROAD",
    "YIELD",
    "STOP",
    "NO ENTRY",
    "NO TRUCKS",
    "NO ENTRY",
    "GENERAL WARNING",
    "LEFT BEND",
    "RIGHT BEND",
    "WINDING ROAD",
    "BUMPY ROAD",
    "SLIPPERY ROAD",
    "ROAD NARROWS",
    "ROADWORKS",
    "TRAFFIC SIGNALS",
    "PEDESTRIAN CROSSING",
    "CHILDREN CROSSING",
    "CYCLISTS",
    "SNOW/ICE WARNING",
    "WILD ANIMALS",
    "END OF ALL RESTRICTIONS",
    "TURN RIGHT",
    "TURN LEFT",
    "GO STRAIGHT",
    "GO STRAIGHT OR RIGHT",
    "GO STRAIGHT OR LEFT",
    "KEEP RIGHT",
    "KEEP LEFT",
    "ROUNDABOUT",
    "END OF NO OVERTAKING",
    "END OF NO OVERTAKING FOR TRUCKS"
]

# --- 4) Define a function to load images from local files or URLs ---
def load_image(source):
    if source.startswith("http"):
        try:
            response = requests.get(source)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            print(f"Loaded image from URL: {source}")
        except Exception as e:
            print(f"Error fetching image from URL: {e}")
            exit(1)
    else:
        try:
            img = Image.open(source).convert('RGB')
            print(f"Loaded local image: {source}")
        except Exception as e:
            print(f"Error opening local image: {e}")
            exit(1)
    return img

# --- 5) Specify the image source (local file path or URL) ---
# Change this value to either a local path or an online URL.
image_source = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFnhQs_XG3u3fe42hzFFjL2Bb-nazAqw-l4w&s"

# --- 6) Load and preprocess the image ---
img = load_image(image_source)

# --- 7) Extract and scale features ---
# Pass a list containing one image so that img2vec returns a 2D array.
features = img2vec.get_vec([img])
scaled_features = scaler.transform(features)

# --- 8) Make a prediction using the classification model ---
predicted_idx = classifier.predict(scaled_features)[0]
predicted_idx = int(predicted_idx)  # Ensure it is an integer index

# Retrieve probabilities if available
if hasattr(classifier, "predict_proba"):
    proba = classifier.predict_proba(scaled_features)[0]
    confidence = proba[predicted_idx]
else:
    proba = None
    confidence = None

predicted_class = class_names[predicted_idx]

# --- 9) Print detection report with aligned class probabilities ---
print("\n=== Detection Report ===")
print("Predicted Class:", predicted_class)
if confidence is not None:
    print("Confidence: {:.2f}".format(confidence))
    print("\nAll class probabilities:")
    for idx, p in enumerate(proba):
        print(f"{idx:2d}: {p:.2f} - {class_names[idx]}")
print("========================\n")

# --- 10) Display the image in a window with overlay text ---
plt.figure(figsize=(8, 8))
plt.imshow(img)
if confidence is not None:
    title_text = f"Predicted: {predicted_class} (Conf: {confidence:.2f})"
else:
    title_text = f"Predicted: {predicted_class}"
plt.title(title_text, fontsize=16)
plt.axis('off')
plt.show()
