

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
from model import create_model, parse_cfg
from utils import output_boxes, draw_predictions
import tensorflow as tf
from config import Config
from datasets.handseg150k.dataset_bboxes import HandsegDatasetBboxes

# disable CUDA, run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_model_size(net_info_block):
    w = int(net_info_block['width'])
    h = int(net_info_block['height'])
    c = int(net_info_block['channels'])
    return (w, h, c)

def test():
    
    blocks = parse_cfg("cfg/yolov3-tiny.cfg")
    net_info = blocks[0]
    model = create_model(blocks)
    # model.load_weights() # when there are any
        
    dataset = HandsegDataset()
    image, mask = dataset.load_image(50050)
    
    image = tf.expand_dims(image, 0)
    image = tf.expand_dims(image, -1)
    image = tf.image.resize(image, [416, 416])
    
    print(image.shape)
    model_size = get_model_size(net_info)
    max_output_size = 5
    max_output_size_per_class=2 # is ignored..
    iou_threshold = 0.5
    confidence_threshold = 0.7
    
    print('Predicting')
    #predictions = model.predict(image)
    predictions = model(image) # better performance for smaller input than predict()
    print(predictions[0].shape)
    # For example, the (1, 460800, 85) shape is returned for 80 classes (85 - 5 = 80)
    # In which format are predictions returned? 
    # box_centers[:,:,0:2], box_shapes[:,:,2:4], confidence[:,:,4:5], classes[:,:,5:]
    print(predictions, predictions[:,:,0:5], predictions[:,:,5:])
    
    print('Calculating boxes')
    boxes, scores, classes, num_detections = output_boxes(predictions, model_size,
        max_output_size, max_output_size_per_class, iou_threshold,
        confidence_threshold)
    
    # shapes...
    # boxes.shape = num_images x num_boxes x four_box_coords
    # scores.shape = num_images x num_boxes
    # classes.shape = num_images x num_boxes
    # num_detections.shape = num_images ; number of valid detections per image
    # ... (1, 2, 4) (1, 2) (1, 2) (1,) 
    # print('Drawing boxes', boxes.shape, scores.shape, classes.shape, num_detections.shape)
    draw_predictions(np.squeeze(image), boxes[0], scores[0], classes[0], num_detections[0])


if __name__ == '__main__':
    test()