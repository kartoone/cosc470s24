import os
from PIL import Image

def resize_images(directory):
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]

    for file in image_files:
        # Open the image file
        image = Image.open(os.path.join(directory, file))

        # Calculate the dimensions for cropping
        width, height = image.size
        if width > height:
            left = (width - height) // 2
            right = left + height
            top = 0
            bottom = height
        else:
            top = (height - width) // 2
            bottom = top + width
            left = 0
            right = width

        # Crop the image to the middle
        image = image.crop((left, top, right, bottom))

        # Resize the image to 32x32 pixels
        image = image.resize((32, 32))

        # Save the resized image
        image.save(os.path.join(directory, f'resized_{file}'))

    print('Image resizing complete.')

# Specify the directory containing the images
directory = 'c:/users/brtoone/Downloads/frames'

# Call the resize_images function
resize_images(directory)