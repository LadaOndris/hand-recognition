

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from datasets.handseg150k.dataset import HandsegDataset, HUGE_INT
from feature_extraction_numba import extract_features, extract_features_and_labels, \
    get_pixel_coords, get_feature_offsets
from joblib import dump, load
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
from skimage.feature import hog
from skimage import data, exposure
from model import Model
from utils import output_boxes, draw_output_boxes
import tensorflow as tf
from config import Config
from datasets.handseg150k.dataset_bboxes import HandsegDatasetBboxes

# disable CUDA, run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def custom_loss():
    
    # xy loss
    
    # wh loss
    
    # confidence loss
    
    # there is no class loss, since there is a single class
    
    
    def loss(y_true, y_pred):
        
        return 1000
    
    
    return loss

def train():
    
    model = Model.from_cfg("cfg/yolov3-tiny.cfg")
    yolo_out_shapes = model.yolo_output_shapes
    tf_model = model.tf_model
    
    config = Config()
    dataset_bboxes = HandsegDatasetBboxes(config)
    dataset_generator = DatasetGenerator(dataset_bboxes.batch_iterator, 
                                         model.input_shape, yolo_out_shapes, model.anchors)
    
    # compile model
    tf_model.compile(optimizer=tf.optimizers.Adam(lr=model.learning_rate), 
                     loss=custom_loss())
        
    tf_model.fit(dataset_generator, epochs=1, verbose=1)
   
    """
    for images, bboxes in dataset.batch_iterator:
        # images.shape is (batch_size, 480, 640, 1)
        # bboxes.shape is (batch_size, 2, 4)
    """
    return

"""
Preprocesses bounding boxes from tf.Dataset and produces y_true.
"""
class DatasetGenerator:
    
    def __init__(self, dataset_bboxes_iterator, input_shape, output_shapes, anchors):
        self.dataset_bboxes_iterator = dataset_bboxes_iterator
        self.strides = self.compute_strides(input_shape, output_shapes)
        self.output_shapes = output_shapes
        self.anchors = anchors
        self.iterator = iter(self.dataset_bboxes_iterator)
        self.n_anchors = 3
    
    def compute_strides(self, input_shape, output_shapes):
        # input_shape is (416, 416, 1)
        grid_sizes = np.array([output_shapes[i][1] 
                               for i in range(len(output_shapes))])
        return input_shape[0] / grid_sizes
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch_images, batch_bboxes = self.iterator.get_next()
        y_true = self.preprocess_true_bboxes(batch_bboxes)
        return batch_images, y_true
    
    
    def preprocess_true_bboxes(self, batch_bboxes):
        y_true = [np.zeros((self.output_shapes[i][0],
                            self.output_shapes[i][1], 
                            self.output_shapes[i][2], 
                            3, 5)) for i in range(len(self.output_shapes))]
        
        for image_in_batch, bboxes in enumerate(batch_bboxes):
            
            # find best anchor for each true bbox
            # (there are 13x13x3 anchors) 
            
            for bbox in bboxes: # for every true bounding box in the image
                # bbox is [x1, y1, x2, y2]
                bbox_center = (bbox[2:] + bbox[:2]) * 0.5
                bbox_wh = bbox[2:] - bbox[:2]
                bbox_xywh = np.concatenate([bbox_center, bbox_wh], axis=-1)
                bbox_xywh_grid_scaled = bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
                
                exist_positive = False
                iou_for_all_scales = []
                
                # for each scale (13x13 and 26x26)
                for scale_index in range(len(self.output_shapes)):
                    
                    anchors_xywh = np.zeros((self.n_anchors, 4))
                    anchors_xywh[:, 0:2] = np.floor(bbox_xywh_grid_scaled[scale_index, 0:2]).astype(np.int32) + 0.5
                    anchors_xywh[:, 2:4] = self.anchors[scale_index]
                    
                    iou_of_this_scale = self.bbox_iou()
                    iou_for_all_scales.append(iou_of_this_scale) 
                    iou_mask = iou_of_this_scale > 0.3
                    
                    if np.any(iou_mask):
                        x_index, y_index = np.floor(bbox_xywh_grid_scaled[scale_index, 0:2]).astype(np.int32)
                        
                        y_true[scale_index][image_in_batch, y_index, x_index, iou_mask, :] = 0
                        y_true[scale_index][image_in_batch, y_index, x_index, iou_mask, 0:4] = bbox_xywh
                        y_true[scale_index][image_in_batch, y_index, x_index, iou_mask, 4:5] = 1.0
                        
                        exist_positive = True
                
                # if no prediction matched the true bounding box enough
                if not exist_positive:
                    pass
                    
                
            
            
            
        #print(len(y_true), y_true[0].shape)
        return None

    

if __name__ == '__main__':
    train()


