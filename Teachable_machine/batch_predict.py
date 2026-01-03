from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os
import glob

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels (this assumes a single model/labels file for all predictions)
model = load_model("keras_model.h5", compile=False)
with open("labels.txt", "r") as f:
    # Assuming labels file may have extra characters; adjust slicing if necessary.
    class_names = [line.strip() for line in f.readlines()]

# Create the "Results" folder if it does not exist
results_folder = "Results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Loop over all folder numbers from 0 to 42
for folder in range(43):
    # Set the image directory path for the current folder
    image_dir = f"C:/Users/sujit/PycharmProjects/Road-Sign-Recognition/Dataset/Train/{folder}"
    # Use glob to collect all .png files in the directory (add more extensions if needed)
    image_files = glob.glob(os.path.join(image_dir, "*.png"))

    # Lists and dictionaries for storing results and statistics for current folder
    results = []
    class_stats = {}
    confidence_stats = {}

    # Process each image file in the current directory
    for image_file in image_files:
        try:
            # Open and convert the image to RGB
            image = Image.open(image_file).convert("RGB")
            # Resize the image to 224x224 using a high-quality downsampling filter and crop from center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            # Convert the image to a NumPy array
            image_array = np.asarray(image)
            # Normalize the image data to match training conditions
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Prepare a data array of the correct shape (1, 224, 224, 3)
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Predict using the loaded model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            confidence_score = prediction[0][index]
            # Slice off any extra characters if needed (adjust this based on your labels format)
            predicted_class = class_names[index][2:]

            # Store the prediction result
            results.append({
                "image_name": os.path.basename(image_file),
                "predicted_class": predicted_class,
                "confidence_score": confidence_score
            })

            # Update class count statistics
            class_stats[predicted_class] = class_stats.get(predicted_class, 0) + 1

            # Collect confidence scores for average calculations later
            if predicted_class in confidence_stats:
                confidence_stats[predicted_class].append(confidence_score)
            else:
                confidence_stats[predicted_class] = [confidence_score]

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Build the summary statistics table as a formatted string
    total_images = len(results)
    summary_table_lines = []
    # Include the image directory path at the very beginning
    summary_table_lines.append(f"Image Directory: {image_dir}")
    summary_table_lines.append("")
    summary_table_lines.append("Summary Statistics")
    summary_table_lines.append("=" * 80)
    summary_table_lines.append(f"Total Images Processed: {total_images}\n")
    summary_table_lines.append("{:<25} {:<10} {:<20}".format("Class", "Count", "Avg Confidence"))
    summary_table_lines.append("-" * 80)
    for cls in class_stats:
        avg_confidence = sum(confidence_stats[cls]) / len(confidence_stats[cls])
        summary_table_lines.append("{:<25} {:<10} {:<20.4f}".format(cls, class_stats[cls], avg_confidence))
    summary_table_lines.append("\n\n")

    # Build the detailed predictions table as a formatted string
    details_table_lines = []
    details_table_lines.append("Detailed Prediction Results")
    details_table_lines.append("=" * 80)
    details_table_lines.append("{:<30} {:<25} {:<20}".format("Image Name", "Predicted Class", "Confidence Score"))
    details_table_lines.append("-" * 80)
    for r in results:
        details_table_lines.append(
            "{:<30} {:<25} {:<20.4f}".format(r['image_name'], r['predicted_class'], r['confidence_score']))
    details_table_lines.append("\n")

    # Combine both tables (with summary statistics placed at the top)
    all_output = "\n".join(summary_table_lines + details_table_lines)

    # Determine the output file name in the "Results" folder.
    # Base filename is based on the folder number.
    base_filename = f"detailed_statistics_{folder}.txt"
    output_file = os.path.join(results_folder, base_filename)
    # If file exists, add numbering to avoid overwriting
    if os.path.exists(output_file):
        counter = 1
        while os.path.exists(os.path.join(results_folder, f"detailed_statistics_{folder}_{counter}.txt")):
            counter += 1
        output_file = os.path.join(results_folder, f"detailed_statistics_{folder}_{counter}.txt")

    # Write the combined output to the text file
    with open(output_file, "w") as f:
        f.write(all_output)

    print(f"Predictions and detailed statistics for folder {folder} have been written to '{output_file}'.")
