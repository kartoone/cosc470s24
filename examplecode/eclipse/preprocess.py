import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.startswith("resized") and (filename.endswith(".jpg") or filename.endswith(".png")):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            #image = image.resize((64, 64))  # Resize the image to desired dimensions - no need to resize since we already have separate script that did that
            image = img_to_array(image)
            images.append(image)
    return np.array(images)

# Specify the directory containing the images
image_directory = "c:/Users/brtoone/Downloads/frames"

# Load images from the directory
dataset = load_images(image_directory)

# Normalize the pixel values to the range [-1, 1]
dataset = (dataset.astype(np.float32) - 127.5) / 127.5

# Print the shape of the dataset
print("Dataset shape:", dataset.shape)