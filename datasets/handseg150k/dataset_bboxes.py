
from PIL import Image
from matplotlib import pyplot as plt
from collections import Counter
from pathlib import Path
import tensorflow as tf
import numpy as np
import os
import glob
import time


# inspired by https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/dataset.py
class HandsegDatasetBboxes:

    def __init__(self, cfg = None, batch_size = 16):
        
        self.dataset_path = os.path.dirname(__file__)
        self.batch_size = batch_size
        self.batch_index = 0
        
        self.annotations = self._load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        
        self.batch_iterator = self._build_iterator()
        return
    
    def _load_annotations(self):
        annotations_path = os.path.join(self.dataset_path, 'bounding_boxes.txt')
        with open(annotations_path, 'r') as f:
            annotations = f.readlines()
        return annotations        
    
    """
    def __iter__(self):
        return self

    
    def __next__(self):
        if self.batch_index >= self.num_batches:
            self.batch_index = 0
            np.random.shuffle(self.annotations)
            raise StopIteration
            
        images_in_batch = 0
        while images_in_batch < self.batch_size:
            image_index = self.batch_index * self.batch_size + images_in_batch
            if image_index < self.num_samples:
                ## load image and bboxes
                ## possibly preproccess image
                pass    
            else:
                break
            images_in_batch += 1
            
        
        self.batch_index += 1
        return None, None
    """ 
    def _build_iterator(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.annotations[:16])
        dataset = dataset.repeat()
        #dataset = dataset.shuffle(len(self.annotations), reshuffle_each_iteration=True)
        dataset = dataset.map(self._prepare_sample)
        shapes = (tf.TensorShape([416, 416, 1]), tf.TensorShape([None, 4]))
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=shapes)
        dataset = dataset.prefetch(buffer_size=5)
        return dataset
    
    # annotation line consists of depth_image file name and bounding boxes coordinates
    @tf.function()
    def _prepare_sample(self, annotation):
        annotation_parts = tf.strings.split(annotation, sep=' ')
        image_file_name = annotation_parts[0]
        image_file_path = tf.strings.join([self.dataset_path, "/images/", image_file_name])
        
        depth_image_file_content = tf.io.read_file(image_file_path)
        depth_image = tf.io.decode_image(depth_image_file_content, channels=1)
        depth_image.set_shape([480, 640, 1]) 
        
        bboxes = tf.reshape(annotation_parts[1:], shape=[-1,4])
        bboxes = tf.strings.to_number(bboxes, out_type=tf.float32)
        
        # resize iamge
        depth_image = tf.image.resize_with_pad(depth_image, 416, 416)
        
        # resize bboxes
        m = tf.tile(tf.constant([[416/640, 416/640, 416/640, 416/640]], dtype=tf.float32), [len(bboxes), 1])
        a = tf.tile(tf.constant([[0, 52, 0, 52]], dtype=tf.float32), [len(bboxes), 1])
        bboxes = tf.math.multiply(bboxes, m)
        bboxes = tf.math.add(bboxes, a)
        
        return depth_image, bboxes
"""
if __name__ == '__main__':
    config = Config()
    dataset = HandsegDatasetBboxes(config)
    
    for image, bboxes in dataset:
        pass
"""