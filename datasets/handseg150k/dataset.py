from PIL import Image
from matplotlib import pyplot as plt
from collections import Counter
from pathlib import Path
import numpy as np
import os
import glob

HUGE_INT = 2147483647


class HandsegDataset:
    
    
    def __init__(self):
        self.image_shape = (480, 640)
        self.offsets = None
        self.pixels = None
        return
    
    def load_images(self, num):
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
    
    def get_random_pixels(self, count, image_shape):
        x = np.random.randint(0, image_shape[0], count, dtype=np.int32) 
        y = np.random.randint(0, image_shape[1], count, dtype=np.int32) 
        return np.column_stack((x, y))
    
    def get_offset(self, count, image_shape):
        half_width = image_shape[0] / 2.0
        half_height = image_shape[1] / 2.0
        x = np.random.randint(-half_width, half_width, count, dtype=np.int32) 
        y = np.random.randint(-half_height, half_height, count, dtype=np.int32) 
        return np.column_stack((x, y))
    
    def get_offsets(self, count, image_shape):
        u = self.get_offset(count, image_shape)
        v = self.get_offset(count, image_shape)
        return np.dstack((u, v))
    
    def get_depth(self, image, coord):
        if (coord[0] >= 0 and coord[1] >= 0 and
            coord[0] < image.shape[0] and
            coord[1] < image.shape[1]):
            return image[coord[0], coord[1]]
        return HUGE_INT
    
    def calculate_feature(self, image, pixel, u, v):
        pixelDepth = image[pixel[0], pixel[1]]#get_depth(image, pixel)
        u = np.divide(u * 10000, pixelDepth).astype(int)
        v = np.divide(v * 10000, pixelDepth).astype(int)
        p1 = np.add(pixel, u)
        p2 = np.add(pixel, v)
        return self.get_depth(image, p1) - self.get_depth(image, p2)
    
    def get_features_for_pixel(self, image, pixel, offsets):
        features = np.array([])
        for u, v in offsets:
            feature = self.calculate_feature(image, pixel, u, v)
            features = np.insert(features, len(features), feature)
            #if feature != 0 and feature < 2147000000 and feature > -2147000000:
            #    print(feature)
            
        return features
    
    def get_label(self, mask, pixel):
        value = mask[pixel[0], pixel[1]]
        if value == 0:
            return 0
        return 1
        
    def get_samples_for_image(self, image, mask, pixels, offsets):
        features = []
        labels = []
        for pixel in pixels:
            pixelFeatures = self.get_features_for_pixel(image, pixel, offsets)
            label = self.get_label(mask, pixel)
            features.append(pixelFeatures)
            labels.append(label)
        return features, labels
            
    """
    Returns a tuple consisting of features and corresponding labels.
    
    Attributes:
        num_images: number of images to extract the features from
        sampled_pixels_count: Number of pixels to be randomly selected and 
            used to extract features from.
            It is not used if pixels parameter is given.
        total_features: Number of feature to be extracted for each pixel.
            It is not used if offsets parameter is given.
        #pixels: List of coordinates, target pixels, to extract the features 
        #    from.
        #offsets: Offsets which are used to calculate features from a pixel. 
        #    For each feature, there is a tuple of two offsets.
    """
    def get_samples(self,
                    num_images = 1, 
                    sampled_pixels_count = 2000, 
                    total_features = 2000):
        images, masks = self.load_images(num_images)
        
        if self.offsets is None:
            self.offsets = self.get_offsets(total_features, self.image_shape)
        if self.pixels is None:
            self.pixels = self.get_random_pixels(sampled_pixels_count, self.image_shape)
        
        features = []
        labels = []
        for image, mask in zip(images, masks):
            f, l = self.get_samples_for_image(image, mask, self.pixels, self.offsets)
            features += f
            labels += l
            
        #print(sum([l for l in labels if l == 1]))
        return np.array(features), np.array(labels)
    
    