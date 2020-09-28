
import cv2
import numpy as np
import matplotlib.pyplot as plt



image_size = (416, 416, 1)
dataset_size = 5000


img = np.zeros(image_size, dtype="uint8")
value = 150
radius_range = (10, 60)

# (dataset_size, 2, 1)
radiuses = np.random.randint(radius_range[0], radius_range[1], size=(dataset_size, 2))
max_window_width = 416 - radiuses

centers = np.random.randint(0, 416, size=(dataset_size, 2, 2))
#print(centers)

#print(max_window_width)
for center, radius in zip(centers, radiuses):
    print(center[0], radius[0])
    cv2.circle(img, tuple(center[0]), radius[0], value, thickness=-1, lineType=8, shift=0)
    cv2.circle(img, tuple(center[1]), radius[1], value, thickness=-1, lineType=8, shift=0)
    break

plt.imshow(img)

