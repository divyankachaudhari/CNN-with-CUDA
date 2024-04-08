import cv2
import numpy as np

# Load the image in grayscale
img = cv2.imread(
    '/Users/divyanka/Assignments/Parallel-PLU-decomposition/Assignment-2/subtask-3/2.png', 0)
if img.shape != [28, 28]:
    img = cv2.resize(img, (28, 28))

# Normalize the image to 0-1 range and invert colors
img_normalized = 1.0 - img / 255.0

# Save the processed image to a file
with open('/Users/divyanka/Assignments/Parallel-PLU-decomposition/Assignment-2/subtask-3/output.txt', 'w') as file:
    for row in img_normalized:
        np.savetxt(file, row[np.newaxis], fmt='%0.6f', delimiter=' ')

# If you prefer a compact format where each line represents a row (optional)
# with open('output_compact.txt', 'w') as file:
#     for row in img_normalized:
#         line = ' '.join(f'{pixel:0.6f}' for pixel in row)
#         file.write(line + '\n')
