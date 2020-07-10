from PIL import Image
from matplotlib import pyplot as plt
from collections import Counter
from pathlib import Path
import numpy as np
import os
import glob

HUGE_INT = 2147483647

def load_images(num):
    dirname = os.path.dirname(__file__)
    all_images_paths = glob.glob(os.path.join(dirname, 'images/*'))
    masks_paths = []
    images_paths = all_images_paths[:num]
    
    for image_path in images_paths:
        _, filename = os.path.split(image_path)
        masks_paths.append(os.path.join(dirname, 'masks', filename))
        
    images = [np.array(Image.open(filename)) for filename in images_paths]
    for image in images:
        image[image == 0] = HUGE_INT
    masks = [np.array(Image.open(filename)) for filename in masks_paths]
    return images, masks
    
"""    
images, masks = load_images(4)

for i, m in zip(images, masks):
    plt.imshow(i); plt.title('Depth Image'); plt.show() # Displaying Depth Image
    plt.imshow(m); plt.title('Mask Image'); plt.show() # Displaying Mask Image
"""    

def get_random_pixels(count, image_shape):
    x = np.random.randint(0, image_shape[0], count, dtype=np.int32) 
    y = np.random.randint(0, image_shape[1], count, dtype=np.int32) 
    return np.column_stack((x, y))

def get_offset(count, image_shape):
    half_width = image_shape[0] / 2.0
    half_height = image_shape[1] / 2.0
    x = np.random.randint(-half_width, half_width, count, dtype=np.int32) 
    y = np.random.randint(-half_height, half_height, count, dtype=np.int32) 
    return np.column_stack((x, y))

def get_offsets(count, image_shape):
    u = get_offset(count, image_shape)
    v = get_offset(count, image_shape)
    return np.dstack((u, v))

def get_depth(image, coord):
    if (coord[0] >= 0 and coord[1] >= 0 and
        coord[0] < image.shape[0] and
        coord[1] < image.shape[1]):
        return image[coord[0], coord[1]]
    return HUGE_INT

def calculate_feature(image, pixel, u, v):
    pixelDepth = image[pixel[0], pixel[1]]#get_depth(image, pixel)
    u = np.divide(u * 10000, pixelDepth).astype(int)
    v = np.divide(v * 10000, pixelDepth).astype(int)
    p1 = np.add(pixel, u)
    p2 = np.add(pixel, v)
    return get_depth(image, p1) - get_depth(image, p2)

def get_features_for_pixel(image, pixel, offsets):
    features = np.array([])
    for u, v in offsets:
        feature = calculate_feature(image, pixel, u, v)
        features = np.insert(features, len(features), feature)
        #if feature != 0 and feature < 2147000000 and feature > -2147000000:
        #    print(feature)
        
    return features
def get_label(mask, pixel):
    value = mask[pixel[0], pixel[1]]
    if value == 0:
        return 0
    return 1
    
def get_samples_for_image(image, mask, pixels, offsets):
    features = []
    labels = []
    for pixel in pixels:
        pixelFeatures = get_features_for_pixel(image, pixel, offsets)
        label = get_label(mask, pixel)
        features.append(pixelFeatures)
        labels.append(label)
    return features, labels
        
"""
Returns a tuple consisting of features and corresponding labels.
"""
def get_samples(num_images = 1, 
                sampled_pixels = 2000, 
                features = 100):
    image_shape = (480, 640)
    images, masks = load_images(num_images)
    offsets = get_offsets(features, image_shape)
    
    features = []
    labels = []
    for image, mask in zip(images, masks):
        pixels = get_random_pixels(sampled_pixels, image_shape)
        f, l = get_samples_for_image(image, mask, pixels, offsets)
        features += f
        labels += l
        
    #print(sum([l for l in labels if l == 1]))
    return np.array(features), np.array(labels)

#f, l = get_samples(1)
#print(f.shape, l.shape)

"""
image = np.ones((10,10))
image[1,1] = 2
image[2,1] = 10
print(calculate_feature(image, 
                        np.array([1,1]), 
                        np.array([2,0]), 
                        np.array([0,2])))
"""
#print(get_offsets(5, (640,480)))    
    
    