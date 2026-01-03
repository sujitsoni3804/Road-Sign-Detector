from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
import os
# at the very top of your script, before any call to load_model()
from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D

class DepthwiseConv2D(_DepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        # drop the unsupported `groups` kwarg if present
        super().__init__(*args, **kwargs)

# … later, in main():
from tensorflow.keras.models import load_model
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# --- 1) Load the model and labels ---
model = load_model(
    "keras_model.h5",
    compile=False,
    custom_objects={'DepthwiseConv2D': DepthwiseConv2D}
)
with open("labels.txt", "r") as f:
    # Adjust slicing if needed depending on the format of your labels.
    class_names = [line.strip() for line in f.readlines()]

# --- 2) Function to load an image from a local file or URL ---
def load_image(source):
    if source.startswith("http"):
        try:
            response = requests.get(source)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            print(f"Loaded image from URL: {source}")
        except Exception as e:
            print(f"Error fetching image from URL: {e}")
            exit(1)
    else:
        try:
            img = Image.open(source).convert("RGB")
            print(f"Loaded local image: {source}")
        except Exception as e:
            print(f"Error opening local image: {e}")
            exit(1)
    return img

# --- 3) Specify the single image source (local file path or URL) ---
# Replace with a local path like "path/to/image.png" or a URL.
image_source = "https://lh3.googleusercontent.com/74KFgBP-eseKmqHdCY0hCOMhTDuqB6n1Dj7CvABA_711g7etlR7gsCnCJMsEOskf4rsIQvA9U94uEBx38poQEzGDcu_z510-vnE65uQmW-zWAVpJScSXKmhd6cjxailVk_sqqRjZu6wXBR5KVsj5mHg"

# --- 4) Load and preprocess the image ---
img = load_image(image_source)
# Resize to 224x224 using a high-quality downsampling filter (with center crop)
size = (224, 224)
img_resized = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
# Convert the image to a NumPy array and normalize it (match training conditions)
image_array = np.asarray(img_resized)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
# Prepare a data array of shape (1, 224, 224, 3)
data = np.expand_dims(normalized_image_array, axis=0)

# --- 5) Predict using the loaded model ---
prediction = model.predict(data)
predicted_index = int(np.argmax(prediction))
confidence_score = prediction[0][predicted_index]
predicted_class = class_names[predicted_index]

# --- 6) Display the result in a window ---
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence_score:.2f}", fontsize=16)
plt.axis("off")
plt.show()







# from keras.models import load_model  # TensorFlow is required for Keras to work
# from PIL import Image, ImageOps  # Install pillow instead of PIL
# import numpy as np
# import os
# import glob

# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)

# # Load the model and labels
# model = load_model("../keras_model.h5", compile=False)
# with open("labels.txt", "r") as f:
#     # Assuming labels file may have extra characters; adjust slicing if necessary.
#     class_names = [line.strip() for line in f.readlines()]

# # Directory containing images
# image_dir = "C:/Users/sujit/PycharmProjects/Road-Sign-Recognition/Dataset/Train/2"
# # Use glob to collect all .png files in the directory. Add more extensions if needed.
# image_files = glob.glob(os.path.join(image_dir, "*.png"))

# # Lists and dictionaries for storing results and statistics
# results = []
# class_stats = {}
# confidence_stats = {}

# # Process each image file in the directory
# for image_file in image_files:
#     try:
#         # Open and convert the image to RGB
#         image = Image.open(image_file).convert("RGB")
#         # Resize the image to 224x224 using a high-quality downsampling filter and crop from center
#         size = (224, 224)
#         image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
#         # Convert the image to a NumPy array
#         image_array = np.asarray(image)
#         # Normalize the image data to match training conditions
#         normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

#         # Prepare a data array of the correct shape (1, 224, 224, 3)
#         data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#         data[0] = normalized_image_array

#         # Predict using the loaded model
#         prediction = model.predict(data)
#         index = np.argmax(prediction)
#         confidence_score = prediction[0][index]
#         # Slice off any extra characters if needed (adjust this based on your labels format)
#         predicted_class = class_names[index][2:]

#         # Store the prediction result
#         results.append({
#             "image_name": os.path.basename(image_file),
#             "predicted_class": predicted_class,
#             "confidence_score": confidence_score
#         })

#         # Update class count statistics
#         class_stats[predicted_class] = class_stats.get(predicted_class, 0) + 1

#         # Collect confidence scores for average calculations later
#         if predicted_class in confidence_stats:
#             confidence_stats[predicted_class].append(confidence_score)
#         else:
#             confidence_stats[predicted_class] = [confidence_score]

#     except Exception as e:
#         print(f"Error processing {image_file}: {e}")

# # Build the summary statistics table as a formatted string
# total_images = len(results)
# summary_table_lines = []
# # Include the image directory path at the very beginning
# summary_table_lines.append(f"Image Directory: {image_dir}")
# summary_table_lines.append("")
# summary_table_lines.append("Summary Statistics")
# summary_table_lines.append("=" * 80)
# summary_table_lines.append(f"Total Images Processed: {total_images}\n")
# summary_table_lines.append("{:<25} {:<10} {:<20}".format("Class", "Count", "Avg Confidence"))
# summary_table_lines.append("-" * 80)
# for cls in class_stats:
#     avg_confidence = sum(confidence_stats[cls]) / len(confidence_stats[cls])
#     summary_table_lines.append("{:<25} {:<10} {:<20.4f}".format(cls, class_stats[cls], avg_confidence))
# summary_table_lines.append("\n\n")

# # Build the detailed predictions table as a formatted string
# details_table_lines = []
# details_table_lines.append("Detailed Prediction Results")
# details_table_lines.append("=" * 80)
# details_table_lines.append("{:<30} {:<25} {:<20}".format("Image Name", "Predicted Class", "Confidence Score"))
# details_table_lines.append("-" * 80)
# for r in results:
#     details_table_lines.append("{:<30} {:<25} {:<20.4f}".format(r['image_name'], r['predicted_class'], r['confidence_score']))
# details_table_lines.append("\n")

# # Combine both tables (with summary statistics placed at the top)
# all_output = "\n".join(summary_table_lines + details_table_lines)

# # Create the "Results" folder if it does not exist
# results_folder = "Results"
# if not os.path.exists(results_folder):
#     os.makedirs(results_folder)

# # Determine the output file name in the "Results" folder.
# base_filename = "detailed_statistics.txt"
# output_file = os.path.join(results_folder, base_filename)
# if os.path.exists(output_file):
#     # If the file exists, add numbering
#     counter = 1
#     while os.path.exists(os.path.join(results_folder, f"detailed_statistics_{counter}.txt")):
#         counter += 1
#     output_file = os.path.join(results_folder, f"detailed_statistics_{counter}.txt")

# # Write the combined output to the text file
# with open(output_file, "w") as f:
#     f.write(all_output)

# print(f"Predictions and detailed statistics have been written to '{output_file}'.")
