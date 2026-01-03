import os
import pickle
import torch
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === 1) Detect GPU availability and init Img2Vec accordingly ===
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("CUDA is available. Using GPU for embeddings.")
else:
    print("CUDA not available. Falling back to CPU.")  # keeps things moving :contentReference[oaicite:1]{index=1}

img2vec = Img2Vec(cuda=use_cuda)

def extract_features(dir_path, batch_size=64):
    """
    Walks through dir_path/{class}/*.jpg, batches images,
    and returns (features, labels) as numpy arrays.
    """
    all_feats, all_labels = [], []
    batch_imgs, batch_labels = [], []

    for category in os.listdir(dir_path):
        cat_folder = os.path.join(dir_path, category)
        for img_name in os.listdir(cat_folder):
            img = Image.open(os.path.join(cat_folder, img_name)).convert('RGB')
            batch_imgs.append(img)
            batch_labels.append(category)

            if len(batch_imgs) == batch_size:
                feats = img2vec.get_vec(batch_imgs)
                all_feats.append(feats)
                all_labels.extend(batch_labels)
                batch_imgs, batch_labels = [], []

    # process any remainder
    if batch_imgs:
        feats = img2vec.get_vec(batch_imgs)
        all_feats.append(feats)
        all_labels.extend(batch_labels)

    X = np.vstack(all_feats)
    y = np.array(all_labels)
    return X, y

# === 2) Load and featurize data ===
train_dir = './Dataset/Train_split'
val_dir   = './Dataset/Val'

print("Extracting train features…")
X_train, y_train = extract_features(train_dir)
print("Extracting val features…")
X_val,   y_val   = extract_features(val_dir)

# === 3) Scale features ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# === 4) Train/dev split for tuning ===
X_tr, X_dev, y_tr, y_dev = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=0
)

# === 5) Hyperparameter tuning ===
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 20, 40],
    'class_weight': [None, 'balanced']
}
rf = RandomForestClassifier(random_state=0, n_jobs=-1)
grid = GridSearchCV(rf, param_grid, cv=3, verbose=1, n_jobs=-1)
grid.fit(X_tr, y_tr)

print("Best RF params:", grid.best_params_)
best_model = grid.best_estimator_

# === 6) Evaluate on dev set ===
y_dev_pred = best_model.predict(X_dev)
print("Dev accuracy:", accuracy_score(y_dev, y_dev_pred))
print(classification_report(y_dev, y_dev_pred))
print("Confusion matrix:\n", confusion_matrix(y_dev, y_dev_pred))

# === 7) Final evaluation on original validation set ===
y_pred = best_model.predict(X_val)
print("Val accuracy:", accuracy_score(y_val, y_pred))

# === 8) Save scaler and model ===
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
