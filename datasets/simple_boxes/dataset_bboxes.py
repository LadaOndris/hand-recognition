import tensorflow as tf
import numpy as np
import os

# inspired by https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/dataset.py
class SimpleBoxesDataset:

    def __init__(self, type, train_size, batch_size = 16):
        if type != 'train' and type != 'test':
            raise ValueError("Invalid dataset type {type}")
        if train_size < 0 or train_size > 1:
            raise ValueError("Train_size expected to be in range [0, 1], but got {train_size}.")
        
        self.dataset_path = os.path.dirname(__file__)
        self.type = type
        self.train_size = train_size
        self.batch_size = batch_size
        self.batch_index = 0
        
        self.annotations = self._load_annotations()
        print("Dataset len", len(self.annotations))
        self.num_samples = len(self.annotations)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        
        self.batch_iterator = self._build_iterator()
        return
    
    def _load_annotations(self):
        annotations_path = os.path.join(self.dataset_path, 'bboxes.txt')
        with open(annotations_path, 'r') as f:
            annotations = f.readlines()
            
        boundary_index = int(len(annotations) * self.train_size)
        if type == 'train':
            return annotations[:boundary_index]
        else:
            return annotations[boundary_index:]
    
    def _build_iterator(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.annotations)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(len(self.annotations), reshuffle_each_iteration=True)
        dataset = dataset.map(self._prepare_sample)
        shapes = (tf.TensorShape([416, 416, 1]), tf.TensorShape([None, 4]))
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=shapes)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset
    
    # annotation line consists of depth_image file name and bounding boxes coordinates
    @tf.function()
    def _prepare_sample(self, annotation):
        annotation_parts = tf.strings.split(annotation, sep=' ')
        image_file_name = annotation_parts[0]
        image_file_path = tf.strings.join([self.dataset_path, "/images/", image_file_name])
        
        depth_image_file_content = tf.io.read_file(image_file_path)
        # loads depth images and converts values to fit in dtype.uint8
        depth_image = tf.io.decode_image(depth_image_file_content, channels=1)
        depth_image.set_shape([416, 416, 1]) 
        #depth_image /= 255 # normalize to range [0, 1]
        
        bboxes = tf.reshape(annotation_parts[1:], shape=[-1,4])
        bboxes = tf.strings.to_number(bboxes, out_type=tf.float32)
        
        return depth_image, bboxes
