# combine_testing.py 
# script to test all 3 different model on specific image
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import cv2
import torch
import pickle
import base64
from PIL import Image, ImageOps
from urllib.parse import urlparse, parse_qs
from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from img2vec_pytorch import Img2Vec

# -------- Image Configuration --------
# Change this to your image URL or local path
IMAGE_SOURCE = "https://www.shutterstock.com/image-photo/german-traffic-sign-maximum-speed-260nw-609632075.jpg"

# -------- Model Paths --------
TEACHABLE_MODEL_PATH = "./Teachable_machine/keras_model.h5"
TEACHABLE_LABELS_PATH = "./Teachable_machine/labels.txt"
YOLO_MODEL_PATH = "./YOLO/runs/classification/yolov8_custom_cls_experiment/weights/best.pt"
SKLEARN_MODEL_PATH = "./Scikit_learn/model.pkl"
SKLEARN_SCALER_PATH = "./Scikit_learn/scaler.pkl"

# -------- Helper Functions --------
class DepthwiseConv2D(_DepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        super().__init__(*args, **kwargs)

def load_image_pil(source):
    if source.startswith("http"):
        try:
            response = requests.get(source)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Error fetching image from URL: {e}")
            exit(1)
    else:
        try:
            img = Image.open(source).convert("RGB")
        except Exception as e:
            print(f"Error opening local image: {e}")
            exit(1)
    return img

def load_image_cv2(source):
    if source.startswith("http") and "imgres" in source:
        parsed = urlparse(source)
        query_params = parse_qs(parsed.query)
        if "imgurl" in query_params:
            source = query_params["imgurl"][0]
        else:
            raise ValueError("The URL does not contain an 'imgurl' parameter.")
    
    if source.startswith("http"):
        response = requests.get(source)
        if response.status_code == 200:
            image_data = response.content
            image_np = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from URL.")
            return img
        else:
            raise ValueError(f"Could not retrieve image from URL. Status code: {response.status_code}")
    elif source.startswith("data:image"):
        header, encoded = source.split(",", 1)
        img_data = base64.b64decode(encoded)
        image_np = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from the data URI.")
        return img
    else:
        img = cv2.imread(source)
        if img is None:
            raise ValueError("Error: Could not load image from local path.")
        return img

# -------- Class Names (Specific for each model) --------
# Teachable Machine and Scikit-learn use the same order
SKLEARN_CLASS_NAMES = [
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

# YOLO has a different order according to its script
YOLO_CLASS_NAMES = [
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

# -------- Model Predictors --------
def predict_teachable_machine(image_source):
    np.set_printoptions(suppress=True)
    
    # Load model
    model = load_model(
        TEACHABLE_MODEL_PATH,
        compile=False,
        custom_objects={'DepthwiseConv2D': DepthwiseConv2D}
    )
    
    # Load labels - if file exists use it, otherwise use standard list
    try:
        with open(TEACHABLE_LABELS_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        class_names = SKLEARN_CLASS_NAMES  # Using the same order as scikit-learn
    
    # Load and preprocess the image
    img = load_image_pil(image_source)
    size = (224, 224)
    img_resized = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(img_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)
    
    # Predict
    prediction = model.predict(data)
    predicted_index = int(np.argmax(prediction))
    confidence_score = float(prediction[0][predicted_index])
    predicted_class = class_names[predicted_index]
    
    return {
        "model": "Teachable Machine",
        "predicted_class": predicted_class,
        "confidence": confidence_score,
        "predicted_index": predicted_index,
        "image": img
    }

def predict_yolo(image_source):
    # Load model
    model = YOLO(YOLO_MODEL_PATH)
    
    # Load and preprocess image
    img = load_image_cv2(image_source)
    
    # Run inference
    results = model.predict(img, verbose=False)
    result = results[0]
    
    # Get prediction
    pred_idx = result.probs.top1
    predicted_class = YOLO_CLASS_NAMES[pred_idx] if pred_idx < len(YOLO_CLASS_NAMES) else f"Class {pred_idx}"
    confidence = result.probs.top1conf.item()
    
    # Convert image for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    return {
        "model": "YOLO",
        "predicted_class": predicted_class,
        "confidence": confidence,
        "predicted_index": pred_idx,
        "image": pil_img
    }

def predict_sklearn(image_source):
    # Initialize Img2Vec
    use_cuda = torch.cuda.is_available()
    img2vec = Img2Vec(cuda=use_cuda)
    
    # Load scaler and model
    with open(SKLEARN_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(SKLEARN_MODEL_PATH, 'rb') as f:
        classifier = pickle.load(f)
    
    # Load and preprocess image
    img = load_image_pil(image_source)
    
    # Extract and scale features
    features = img2vec.get_vec([img])
    scaled_features = scaler.transform(features)
    
    # Make prediction
    predicted_idx = classifier.predict(scaled_features)[0]
    predicted_idx = int(predicted_idx)
    
    # Get confidence if available
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(scaled_features)[0]
        confidence = proba[predicted_idx]
    else:
        confidence = None
    
    predicted_class = SKLEARN_CLASS_NAMES[predicted_idx]
    
    return {
        "model": "Scikit-Learn",
        "predicted_class": predicted_class,
        "confidence": confidence,
        "predicted_index": predicted_idx,
        "image": img
    }

# -------- Main Execution --------
def run_comparison(image_source):
    results = []
    
    # Run each model and collect results
    try:
        print("Running Teachable Machine model...")
        teachable_result = predict_teachable_machine(image_source)
        results.append(teachable_result)
    except Exception as e:
        print(f"Error with Teachable Machine model: {e}")
    
    try:
        print("Running YOLO model...")
        yolo_result = predict_yolo(image_source)
        results.append(yolo_result)
    except Exception as e:
        print(f"Error with YOLO model: {e}")
    
    try:
        print("Running Scikit-Learn model...")
        sklearn_result = predict_sklearn(image_source)
        results.append(sklearn_result)
    except Exception as e:
        print(f"Error with Scikit-Learn model: {e}")
    
    # Display comparison results
    if results:
        # Print tabular comparison
        print("\n===== MODEL COMPARISON RESULTS =====")
        print(f"{'Model':<20} {'Prediction':<30} {'Confidence':<10}")
        print("-" * 60)
        
        for result in results:
            confidence_str = f"{result['confidence']:.4f}" if result['confidence'] is not None else "N/A"
            print(f"{result['model']:<20} {result['predicted_class']:<30} {confidence_str:<10}")
        
        # Get agreements and disagreements
        predictions = [r["predicted_class"] for r in results]
        unique_predictions = set(predictions)
        
        if len(unique_predictions) == 1:
            print("\nAll models agree on prediction:", predictions[0])
        else:
            print("\nModels have different predictions.")
            
            # Count agreements to determine the most common prediction
            prediction_counts = {}
            for pred in predictions:
                if pred in prediction_counts:
                    prediction_counts[pred] += 1
                else:
                    prediction_counts[pred] = 1
            
            # Find the most common prediction(s)
            max_count = max(prediction_counts.values())
            most_common = [p for p, c in prediction_counts.items() if c == max_count]
            
            if len(most_common) == 1 and max_count > 1:
                print(f"Majority prediction: {most_common[0]} ({max_count}/{len(results)} models)")
            else:
                print("No majority agreement among models")
        
        # Display images with predictions
        fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
        if len(results) == 1:
            axes = [axes]  # Handle single plot case
        
        for i, result in enumerate(results):
            axes[i].imshow(result["image"])
            conf_str = f" (Conf: {result['confidence']:.2f})" if result['confidence'] is not None else ""
            axes[i].set_title(f"{result['model']}\n{result['predicted_class']}{conf_str}")
            axes[i].axis("off")
        
        plt.tight_layout()
        plt.show()
    else:
        print("No model predictions were successful.")

if __name__ == "__main__":
    print(f"Analyzing image: {IMAGE_SOURCE}")
    run_comparison(IMAGE_SOURCE)