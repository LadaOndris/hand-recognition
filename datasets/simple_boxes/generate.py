
import os
import cv2
import numpy as np

"""
Run this file to generate the simple dataset.

Edit the dataset_size to determine the number of images in the dataset.
"""

current_dir_path = os.path.dirname(__file__)
images_path = os.path.join(current_dir_path, "images")
os.makedirs(images_path, exist_ok=True)

image_size = (416, 416, 1)
dataset_size = 10

value = 150
radius_range = (10, 60)

# (dataset_size, 2, 1)
radiuses = np.random.randint(radius_range[0], radius_range[1], size=(dataset_size, 2, 1))
max_window_width = 416 - radiuses

centers = np.random.randint(0, 416, size=(dataset_size, 2, 2))

 # move boxes which are partly out of the image
centers = np.where(centers < radiuses, centers + radiuses, centers)
centers = np.where(centers > 416 - radiuses, centers - radiuses, centers)



lines = []
for i, (center, radius) in enumerate(zip(centers, radiuses)):
    topleft = center - radius
    rightbottom = center + radius
    
    img = np.zeros(image_size, dtype="uint8")
    cv2.circle(img, tuple(center[0]), radius[0], value, thickness=-1, lineType=8, shift=0)
    cv2.circle(img, tuple(center[1]), radius[1], value, thickness=-1, lineType=8, shift=0)
    file_name = F"image_{i}.png"
    cv2.imwrite(os.path.join(images_path, file_name), img)
    
    lines.append(file_name)
    lines[i] += F" {topleft[0][0]} {topleft[0][1]} {rightbottom[0][0]} {rightbottom[0][1]}"
    lines[i] += F" {topleft[1][0]} {topleft[1][1]} {rightbottom[1][0]} {rightbottom[1][1]}"
    lines[i] += '\n'
    
    

with open(os.path.join(current_dir_path, "bboxes.txt"), 'w') as f:
    f.writelines(lines)

