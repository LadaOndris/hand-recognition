from src.datasets.handseg150k.dataset import HUGE_INT
from numba import jit
import numpy as np

def get_pixel_coords(feature_pixel_distance, image_shape):
    return np.array([[i, j] for i in range(0, image_shape[0], feature_pixel_distance) 
                    for j in range(0, image_shape[1], feature_pixel_distance)])
    

def get_offset(count, image_shape):
    half_width = image_shape[0] / 2.0
    half_height = image_shape[1] / 2.0
    x = np.random.randint(-half_width, half_width, count, dtype=np.int32) 
    y = np.random.randint(-half_height, half_height, count, dtype=np.int32) 
    return np.column_stack((x, y))

def get_feature_offsets(count, image_shape):
    u = get_offset(count, image_shape)
    v = get_offset(count, image_shape)
    return np.dstack((u, v))

@jit(nopython=True)
def get_depth_m(image, coords):
    depths = np.full(shape=(coords.shape[0], coords.shape[1]), fill_value=HUGE_INT, dtype=np.int32)
    for j, coords_for_pixel in enumerate(coords):
        for i, c in enumerate(coords_for_pixel):
            if (c[0] >= 0 and c[1] >= 0 and
                c[0] < image.shape[0] and
                c[1] < image.shape[1]):
                depths[j, i] = image[c[0], c[1]]
    return depths

"""
def get_features_for_pixel_m(image, pixel, u, v):
    pixelDepth = image[pixel[0], pixel[1]]
    u = np.divide(u * 10000, pixelDepth).astype(np.int32)
    v = np.divide(v * 10000, pixelDepth).astype(np.int32)
    p1 = np.add(pixel, u)
    p2 = np.add(pixel, v)
    return np.subtract(get_depth_m(image, p1), get_depth_m(image, p2))
"""
def get_features_for_pixels_m(image, pixels, u, v):
    pixelDepths = image[pixels[:,0], pixels[:,1]]
    u = np.divide(u * 10000, pixelDepths[:, np.newaxis, np.newaxis]).astype(np.int32)
    v = np.divide(v * 10000, pixelDepths[:, np.newaxis, np.newaxis]).astype(np.int32)
    p1 = np.add(pixels[:, np.newaxis, :], u)
    p2 = np.add(pixels[:, np.newaxis, :], v)
    return np.subtract(get_depth_m(image, p1), get_depth_m(image, p2))
    
def get_label(mask, pixel):
    value = mask[pixel[0], pixel[1]]
    if value == 0:
        return 0
    return 1   
"""
Extracts features for all given images.
The output features shape is (images * sampled_image_pixels, features_per_pixel).

"""
def extract_features(images, 
                     sampled_pixels_distance = 12, 
                     features_per_pixel = 2000, 
                     offsets = None, pixels = None): 
    image_shape = images[0].shape
    
    if offsets is None:
        offsets = get_feature_offsets(features_per_pixel, image_shape)
    if pixels is None:
        pixels = get_pixel_coords(sampled_pixels_distance, image_shape)
        
    num_images = len(images)
    features = np.ndarray(shape=(num_images * len(pixels), len(offsets)))
        
    u = offsets[:,0]
    v = offsets[:,1]
    
    for i, image in enumerate(images):
        index = i * len(pixels)
        features[index : index + len(pixels)] = get_features_for_pixels_m(image, pixels, u, v)
    return features
"""
Extracts features for all given images.
The output features shape is (images * sampled_image_pixels, features_per_pixel).
The output labels shape is (images * sampled_image_pixels,).
"""
def extract_features_and_labels(images, masks, 
                                sampled_pixels_distance = 12, 
                                features_per_pixel = 2000, 
                                offsets = None, pixels = None):
    image_shape = images[0].shape
    
    if offsets is None:
        offsets = get_feature_offsets(features_per_pixel, image_shape)
    if pixels is None:
        pixels = get_pixel_coords(sampled_pixels_distance, image_shape)
        
    num_images = len(images)
    features = np.ndarray(shape=(num_images * len(pixels), len(offsets)))
    labels = np.ndarray(shape=(num_images * len(pixels)))
        
    u = offsets[:,0]
    v = offsets[:,1]
    for i, (image, mask) in enumerate(zip(images, masks)):
        index = i * len(pixels)
        features[index : index + len(pixels)] = get_features_for_pixels_m(image, pixels, u, v)
        for p, pixel in enumerate(pixels):
            labels[index + p] = get_label(mask, pixel)
            
    return features, labels