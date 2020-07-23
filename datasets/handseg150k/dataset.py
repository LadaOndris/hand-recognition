from PIL import Image
from matplotlib import pyplot as plt
from collections import Counter
from pathlib import Path
import numpy as np
import os
import glob
import time

HUGE_INT = 2147483647


class HandsegDataset:
    
    
    def __init__(self):
        self.image_shape = (480, 640)
        self.offsets = None
        self.pixels = None
        return
    
    """
    Load a subset of images of the dataset.
    The subset is defined by start_index and end_index.
    
    If start_index is set to None, the subset starts at the beginning of the dataset.
    If end_index is set to None, the subset ends at the end of the dataset.
    If both set to None, selects the whole dataset. Not recommended.
    """
    def load_images(self, start_index = 0, end_index = 99):
        dirname = os.path.dirname(__file__)
        all_images_paths = glob.glob(os.path.join(dirname, 'images/*'))
        masks_paths = []
        
        if (start_index != None and end_index != None):
            images_paths = all_images_paths[start_index:end_index]
        elif (start_index == None and end_index != None):
            images_paths = all_images_paths[:end_index]
        elif (start_index != None and end_index == None):
            images_paths = all_images_paths[start_index:]
        else:
            images_paths = all_images_paths
        
        for image_path in images_paths:
            _, filename = os.path.split(image_path)
            masks_paths.append(os.path.join(dirname, 'masks', filename))
            
        images = np.array([np.array(Image.open(filename)) for filename in images_paths])
        for image in images:
            image[image == 0] = HUGE_INT
        masks = [np.array(Image.open(filename)) for filename in masks_paths]
        return images, masks
        
    """
    Reads a depth image and its mask at the specified index.
    """
    def load_image(self, index):
        dirname = os.path.dirname(__file__)
        all_images_paths = glob.glob(os.path.join(dirname, 'images/*'))
        image_path = all_images_paths[index]
        
        _, filename = os.path.split(image_path)
        mask_path = os.path.join(dirname, 'masks', filename)
            
        image = np.array(Image.open(image_path))
        image[image == 0] = HUGE_INT
        mask = np.array(Image.open(mask_path))
        return image, mask
        
        
    """    
    images, masks = load_images(4)
    
    for i, m in zip(images, masks):
        plt.imshow(i); plt.title('Depth Image'); plt.show() # Displaying Depth Image
        plt.imshow(m); plt.title('Mask Image'); plt.show() # Displaying Mask Image
    """    
    
    def get_random_pixels(self, count, image_shape):
        #x = np.arange(0, 480, step=12)
        #y = np.arange(0, 640, step=12)
        return [[i, j] for i in range(0, 480, 12) for j in range(0, 640, 12)]
                
        #mx, my = np.meshgrid(x, y)
        #print(mx.shape, my.shape)
        
        #x = np.random.randint(0, image_shape[0], count, dtype=np.int32) 
        #y = np.random.randint(0, image_shape[1], count, dtype=np.int32) 
        
        #return np.column_stack((x, y))
    
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
    
    def get_depth_m(self, image, coords):
        depths = np.full(shape=(len(coords)), fill_value=HUGE_INT, dtype=int)
        """
        for i, c in enumerate(coords):
            if (c[0] >= 0 and c[1] >= 0 and
                c[0] < image.shape[0] and
                c[1] < image.shape[1]):
                depths[i] = image[c[0], c[1]]
        """ 
        """
        x_coords = coords[:,0]
        y_coords = coords[:,1]
        x_coords[x_coords < 0] = 0
        y_coords[y_coords < 0] = 0
        x_coords[x_coords >= image.shape[0]] = 0
        y_coords[y_coords >= image.shape[1]] = 0
        depths = image[x_coords, y_coords]
        """
        
        mask = (coords[:,0] < image.shape[0]) & (coords[:,1] < image.shape[1]) & \
            (coords[:,0] >= 0) & (coords[:,1] >= 0)
        valid = coords[mask]
        depths[mask] = image[valid[:, 0], valid[:, 1]]
        
        #coords = coords
                          #x_coords < image.shape[0] and
                          #y_coords >= 0 and
                          #y_coords < image.shape[1]]
        #print(depths, "###")
        
        return depths
    
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
    
    def get_features_for_pixel_m(self, image, pixel):
        u = self.offsets[:,0]
        v = self.offsets[:,1]
        pixelDepth = image[pixel[0], pixel[1]]
        u = np.divide(u * 10000, pixelDepth).astype(int)
        v = np.divide(v * 10000, pixelDepth).astype(int)
        p1 = np.add(pixel, u)
        p2 = np.add(pixel, v)
        return np.subtract(self.get_depth_m(image, p1), self.get_depth_m(image, p2))
        
    
    def get_label(self, mask, pixel):
        value = mask[pixel[0], pixel[1]]
        if value == 0:
            return 0
        return 1
        
    """
    Computes features and labels for the given image.
    """
    def get_samples_for_image(self, image, mask):
        features = np.empty(shape=(len(self.pixels), len(self.offsets)), dtype=int)
        labels = np.empty(shape=(len(self.pixels)), dtype=int)
        for i, pixel in enumerate(self.pixels):
            features[i] = self.get_features_for_pixel_m(image, pixel)
            labels[i] = self.get_label(mask, pixel)
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
                    image_start_index = 0,
                    image_end_index = 99,
                    sampled_pixels_count = 2000, 
                    total_features = 2000):
        images, masks = self.load_images(image_start_index, image_end_index)
        
        if self.offsets is None:
            self.offsets = self.get_offsets(total_features, self.image_shape)
        if self.pixels is None:
            self.pixels = self.get_random_pixels(sampled_pixels_count, self.image_shape)
        
        num_images = image_end_index - image_start_index + 1
        features = np.ndarray(shape=(num_images * len(self.pixels), len(self.offsets)))
        labels = np.ndarray(shape=(num_images * len(self.pixels)))
        
        for i, (image, mask) in enumerate(zip(images, masks)):
            for p, pixel in enumerate(self.pixels):
                #if image[pixel[0], pixel[1]] == HUGE_INT:
                #    continue
                features[i * len(self.pixels) + p] = self.get_features_for_pixel_m(image, pixel)
                labels[i * len(self.pixels) + p] = self.get_label(mask, pixel)
            
        #print(sum([l for l in labels if l == 1]))
        return features, labels

"""
d = HandsegDataset()
start_time = time.monotonic()
s = d.get_samples(10)
print('seconds:', (time.monotonic() - start_time))
print(s[0].shape, s[1].shape)
#print([a  for i in range(len(s[0])) for a in s[0][i] if a != 0 and a < 2147000000 and a > -2147000000])
"""