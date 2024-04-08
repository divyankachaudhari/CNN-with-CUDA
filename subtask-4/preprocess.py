import cv2
import numpy as np
import os
import glob


def process_image(image_path, output_dir):
    # Load the image in grayscale
    img = cv2.imread(image_path, 0)

    # Resize the image to 28x28 if it's not already
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))

    # Normalize the image to 0-1 range and invert colors
    img_normalized = img / 255.0

    # Construct the output file path
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename.replace('.png', '.txt'))

    # Save the processed image data to a file
    with open(output_path, 'w') as file:
        for row in img_normalized:
            np.savetxt(file, row[np.newaxis], fmt='%0.6f', delimiter=' ')

    print(f'Processed {image_path} and saved to {output_path}')


def process_all_images(source_dir, output_dir):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all .png images in the source directory
    image_paths = glob.glob(os.path.join(source_dir, '*.png'))

    # Process each image
    for image_path in image_paths:
        process_image(image_path, output_dir)

    print("All images have been processed.")


# Example usage
source_directory = '/Users/divyanka/Assignments/data'
output_directory = '/Users/divyanka/Assignments/processed_data'
process_all_images(source_directory, output_directory)
