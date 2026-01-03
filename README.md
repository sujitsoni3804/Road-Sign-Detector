# Road Sign Recognition Project

## Project Overview

This project involves the development and comparison of three different machine learning models for road sign recognition:

1. **Scikit-Learn**: A traditional machine learning approach using image embeddings and a RandomForest classifier
2. **YOLOv8**: An end-to-end deep learning approach using YOLOv8 in classification mode
3. **Teachable Machine**: A rapid prototype created using Google's Teachable Machine web interface

The project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains images of 43 different road signs.

## Project Structure

```
├── Dataset/
│   ├── Train_split/     # 70% of data
│   ├── Val/             # 15% of data
│   └── Test/            # 15% of data
├── Scikit_learn/
│   ├── scikit_learn_model.py
│   ├── model.pkl        # Trained model
│   └── scaler.pkl       # Feature scaler
├── YOLO/
│   ├── yolo_model.py
│   ├── gtsrb.yaml       # Dataset configuration
│   └── runs/            # Training outputs
├── Teachable_machine/
│   ├── keras_model.h5   # Exported model
│   └── labels.txt       # Class labels
├── combine_testing.py   # Comparative testing script
└── README.md
```

## Installation & Dependencies

```bash
# Core dependencies
pip install numpy matplotlib requests opencv-python torch pillow tensorflow

# Model-specific dependencies
pip install scikit-learn img2vec-pytorch ultralytics
```

## Model Approaches

### 1. Scikit-Learn Model

This approach uses feature extraction via the img2vec-pytorch library (which leverages a pretrained CNN) followed by a RandomForest classifier:

- **Feature Extraction**: Images are processed through a pretrained CNN to extract meaningful feature vectors
- **Model**: RandomForest with hyperparameter tuning via GridSearchCV
- **Preprocessing**: StandardScaler for feature normalization

```bash
python scikit_learn_model.py
```

### 2. YOLOv8 Classification Model

YOLOv8 is used in classification mode (rather than object detection) to recognize road signs:

- **Base Model**: yolov8n-cls (the nano version optimized for classification)
- **Training**: 50 epochs with SGD optimizer
- **Input Size**: 288×288 pixels

```bash
python yolo_model.py
```

### 3. Teachable Machine Model

Google's Teachable Machine was used to quickly create a model without coding:

- Web interface used to upload and categorize images
- TensorFlow/Keras model exported
- Simple transfer learning approach using a MobileNet backbone

## Comparative Analysis

The `combine_testing.py` script provides a unified interface to test all three models on the same image and compare results.

```bash
# Modify the IMAGE_SOURCE variable in the script to test with your own images
python combine_testing.py
```

The script:
- Loads each model
- Processes the input image for each model's requirements
- Displays predictions side-by-side with confidence scores
- Highlights agreements and disagreements between models

## Results & Conclusions

After extensive testing across the validation set and with unseen images, we found:

1. **Performance Ranking**:
   - **YOLOv8** performed best on both test data and unseen images
   - **Scikit-Learn** model came in second
   - **Teachable Machine** model showed the lowest accuracy

2. **Confidence Calibration**:
   - All models showed suboptimal confidence calibration
   - Models often displayed high confidence even when predictions were incorrect
   - YOLOv8 had the best-calibrated confidence scores, but still showed overconfidence in some cases

3. **Processing Speed**:
   - YOLOv8 was the fastest for inference once loaded
   - Scikit-Learn had the overhead of feature extraction
   - Teachable Machine had reasonable performance but required more preprocessing

4. **Model Size & Complexity**:
   - YOLOv8: Largest model (~20MB)
   - Teachable Machine: Medium size (~8MB)
   - Scikit-Learn: Smallest total size but requires feature extractor

5. **Edge Cases**:
   - Low light conditions and partial occlusions were challenging for all models
   - YOLOv8 showed higher robustness to variations in viewing angle and distance

## Future Improvements

- Implement confidence calibration techniques
- Add data augmentation to improve model robustness
- Explore ensemble methods combining the strengths of different approaches
- Test with more diverse and challenging datasets
- Optimize models for edge/mobile deployment
