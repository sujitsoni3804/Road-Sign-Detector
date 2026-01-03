import os
import pickle
import torch
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1) Load the saved model and scaler ---
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- 2) Setup Img2Vec (using GPU if available) ---
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("CUDA is available. Using GPU for embeddings.")
else:
    print("CUDA not available. Falling back to CPU.")
img2vec = Img2Vec(cuda=use_cuda)

# --- 3) Define class mapping ---
class_names = {
    0: "SPEED LIMIT 20",
    1: "SPEED LIMIT 30",
    2: "SPEED LIMIT 50",
    3: "SPEED LIMIT 60",
    4: "SPEED LIMIT 70",
    5: "SPEED LIMIT 80",
    6: "END OF SPEED LIMIT 80",
    7: "SPEED LIMIT 100",
    8: "SPEED LIMIT 120",
    9: "NO OVERTAKING",
    10: "NO OVERTAKING BY TRUCKS",
    11: "INTERSECTION WARNING",
    12: "PRIORITY ROAD",
    13: "YIELD",
    14: "STOP",
    15: "NO ENTRY",
    16: "NO TRUCKS",
    17: "NO ENTRY",
    18: "GENERAL WARNING",
    19: "LEFT BEND",
    20: "RIGHT BEND",
    21: "WINDING ROAD",
    22: "BUMPY ROAD",
    23: "SLIPPERY ROAD",
    24: "ROAD NARROWS",
    25: "ROADWORKS",
    26: "TRAFFIC SIGNALS",
    27: "PEDESTRIAN CROSSING",
    28: "CHILDREN CROSSING",
    29: "CYCLISTS",
    30: "SNOW/ICE WARNING",
    31: "WILD ANIMALS",
    32: "END OF ALL RESTRICTIONS",
    33: "TURN RIGHT",
    34: "TURN LEFT",
    35: "GO STRAIGHT",
    36: "GO STRAIGHT OR RIGHT",
    37: "GO STRAIGHT OR LEFT",
    38: "KEEP RIGHT",
    39: "KEEP LEFT",
    40: "ROUNDABOUT",
    41: "END OF NO OVERTAKING",
    42: "END OF NO OVERTAKING FOR TRUCKS"
}

# --- 4) Define a function to extract features from a directory ---
def extract_features(dir_path, batch_size=64):
    """
    Walk through dir_path/{class}/*.jpg, batch process images,
    and return (features, labels) where labels are strings.
    """
    all_feats, all_labels = [], []
    batch_imgs, batch_labels = [], []

    # Sorted by numeric value to ensure consistent ordering
    for category in sorted(os.listdir(dir_path), key=lambda x: int(x)):
        cat_folder = os.path.join(dir_path, category)
        if not os.path.isdir(cat_folder):
            continue
        for img_name in tqdm(os.listdir(cat_folder), desc=f"Processing class {category}"):
            img_path = os.path.join(cat_folder, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
            batch_imgs.append(img)
            # Keep the label as string for consistency with training
            batch_labels.append(category)
            if len(batch_imgs) == batch_size:
                feats = img2vec.get_vec(batch_imgs)
                all_feats.append(feats)
                all_labels.extend(batch_labels)
                batch_imgs, batch_labels = [], []
    # Process any remaining images
    if batch_imgs:
        feats = img2vec.get_vec(batch_imgs)
        all_feats.append(feats)
        all_labels.extend(batch_labels)

    X = np.vstack(all_feats)
    y = np.array(all_labels)
    return X, y

# --- 5) Extract features from the test dataset ---
test_dir = './Dataset/Test'
print("Extracting test features …")
X_test, y_test = extract_features(test_dir)

# --- 6) Scale the features using the loaded scaler ---
X_test = scaler.transform(X_test)

# --- 7) Make predictions with the model ---
y_pred = model.predict(X_test)

# --- 8) Evaluation: print accuracy, classification report, and confusion matrix ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}\n")

report = classification_report(
    y_test,
    y_pred,
    target_names=[class_names[int(i)] for i in sorted(np.unique(y_test), key=int)]
)
print("Classification Report:")
print(report)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# --- 9) Save the classification report to a text file ---
output_file = "classification_report.txt"
with open(output_file, "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(np.array2string(cm))
    
print(f"\nThe classification report and confusion matrix have been saved to '{output_file}'.")
