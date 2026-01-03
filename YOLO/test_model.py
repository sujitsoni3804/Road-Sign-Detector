import cv2
import matplotlib.pyplot as plt
import requests
import base64
import numpy as np
from urllib.parse import urlparse, parse_qs
from ultralytics import YOLO

# ---------------------------
# Configuration
# ---------------------------
# model_path = 'runs/classification/yolov8_custom_cls_experiment/weights/best.pt'  # Trained YOLOv8 weights
model_path = 'best.pt'  # Trained YOLOv8 weights

# You can change the image_source value to either:
# 1. A direct image URL (make sure it is a direct link to the image, not a search result page)
# 2. A data URI (base64 encoded image)
# 3. A local file path
#
# For example, using a Google image search URL:
image_source = ("https://lh3.googleusercontent.com/74KFgBP-eseKmqHdCY0hCOMhTDuqB6n1Dj7CvABA_711g7etlR7gsCnCJMsEOskf4rsIQvA9U94uEBx38poQEzGDcu_z510-vnE65uQmW-zWAVpJScSXKmhd6cjxailVk_sqqRjZu6wXBR5KVsj5mHg")

# Class names based on the YAML file configuration (0 to 42)
class_names = [
    "SPEED LIMIT 20",              # '0'
    "SPEED LIMIT 30",              # '1'
    "NO OVERTAKING BY TRUCKS",     # '10'
    "INTERSECTION WARNING",        # '11'
    "PRIORITY ROAD",               # '12'
    "YIELD",                       # '13'
    "STOP",                        # '14'
    "NO ENTRY",                    # '15'
    "NO TRUCKS",                   # '16'
    "NO ENTRY",                    # '17'
    "GENERAL WARNING",             # '18'
    "LEFT BEND",                   # '19'
    "SPEED LIMIT 50",              # '2'
    "RIGHT BEND",                  # '20'
    "WINDING ROAD",                # '21'
    "BUMPY ROAD",                  # '22'
    "SLIPPERY ROAD",               # '23'
    "ROAD NARROWS",                # '24'
    "ROADWORKS",                   # '25'
    "TRAFFIC SIGNALS",             # '26'
    "PEDESTRIAN CROSSING",         # '27'
    "CHILDREN CROSSING",           # '28'
    "CYCLISTS",                    # '29'
    "SPEED LIMIT 60",              # '3'
    "SNOW/ICE WARNING",            # '30'
    "WILD ANIMALS",                # '31'
    "END OF ALL RESTRICTIONS",     # '32'
    "TURN RIGHT",                  # '33'
    "TURN LEFT",                   # '34'
    "GO STRAIGHT",                 # '35'
    "GO STRAIGHT OR RIGHT",        # '36'
    "GO STRAIGHT OR LEFT",         # '37'
    "KEEP RIGHT",                  # '38'
    "KEEP LEFT",                   # '39'
    "SPEED LIMIT 70",              # '4'
    "ROUNDABOUT",                  # '40'
    "END OF NO OVERTAKING",        # '41'
    "END OF NO OVERTAKING FOR TRUCKS",  # '42'
    "SPEED LIMIT 80",              # '5'
    "END OF SPEED LIMIT 80",       # '6'
    "SPEED LIMIT 100",             # '7'
    "SPEED LIMIT 120",             # '8'
    "NO OVERTAKING"                # '9'
]

# ---------------------------
# Helper Function to Load Image
# ---------------------------
def load_image(source):
    """
    Load an image from a local file, HTTP URL, or a data URI (base64 string).
    
    Parameters:
        source (str): Local path, URL, or data URI.
    
    Returns:
        img (numpy.ndarray): Loaded image in BGR format.
    """
    # If the URL is from Google image search, extract the direct image URL from the "imgurl" parameter.
    if source.startswith("http") and "imgres" in source:
        parsed = urlparse(source)
        query_params = parse_qs(parsed.query)
        if "imgurl" in query_params:
            source = query_params["imgurl"][0]  # Use the first occurrence of imgurl parameter.
        else:
            raise ValueError("The URL does not contain an 'imgurl' parameter.")
    
    if source.startswith("http"):
        # Direct URL: Download the image using requests.
        response = requests.get(source)
        if response.status_code == 200:
            image_data = response.content
            image_np = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from URL.")
            return img
        else:
            raise ValueError("Could not retrieve image from URL. Status code: {}".format(response.status_code))
    elif source.startswith("data:image"):
        # Data URI: Strip the header and decode the base64 string.
        header, encoded = source.split(",", 1)
        img_data = base64.b64decode(encoded)
        image_np = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from the data URI.")
        return img
    else:
        # Assume it is a local file path.
        img = cv2.imread(source)
        if img is None:
            raise ValueError("Error: Could not load image from local path.")
        return img

# ---------------------------
# Load the Model
# ---------------------------
print("Loading model...")
model = YOLO(model_path)
print("Model loaded successfully.")

# ---------------------------
# Load the Image (from URL, Data URI, or Local File)
# ---------------------------
try:
    img = load_image(image_source)
except Exception as e:
    print(e)
    exit(1)

# ---------------------------
# Run Inference
# ---------------------------
results = model.predict(img, verbose=False)
result = results[0]

# Use the provided attribute `top1` for the highest-probability prediction
pred_idx = result.probs.top1  # Index of the predicted class
predicted_class = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
confidence = result.probs.top1conf.item()  # Get confidence as a float

# ---------------------------
# Output the Prediction
# ---------------------------
print("Predicted class:", predicted_class)
print("Confidence:", confidence)

# ---------------------------
# Display the Image (at Original Size)
# ---------------------------
# Convert from BGR (OpenCV) to RGB (matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img_rgb.shape[:2]

# Set DPI and figure size
dpi = 100  # You can increase this value for higher resolution
min_width, min_height = 800, 600  # Minimum width and height in pixels
fig_width = max(width, min_width) / dpi
fig_height = max(height, min_height) / dpi

plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
plt.imshow(img_rgb)
plt.title(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
plt.axis("off")
plt.tight_layout()
plt.show()