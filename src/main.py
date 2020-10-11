"""
1. detection
2. pose estimation
3. gesture fulfillment
"""

#from detection.rdf import load_model
import detection
import pose_estimation
import matplotlib.pyplot as plt  
import pyrealsense2 as rs
import numpy as np
import tensorflow as tf
import sys
import os
from PIL import Image 
from src.detection.yolov3 import utils
from src.detection.yolov3 import model as yolov3_model

"""
def detect_and_estimate(depth_im):
    detection_model = detection.rdf.load_model('', '')
    bounding_box = detection.predict_boundary(detection_model, depth_im)
    pose_model = pose_estimation.deep_prior.load_model('', '')
    pose = pose_estimation.deep_prior.predict_pose(pose_model, bounding_box)
"""
   
def print_live_images(num = None):
    generator = generate_live_images()
    
    i = 0
    while True:
        if i == num:
            break
        i += 1
        
        depth_image = next(generator)
        plt.imshow(depth_image)
        plt.show()
        
def generate_live_images():
    pipe = rs.pipeline()
    profile = pipe.start()
    try:
      while True:
        frameset = pipe.wait_for_frames()
        depth_frame = frameset.get_depth_frame()
        depth_image = np.array(depth_frame.get_data())
        depth_image = depth_image[..., np.newaxis]
        
        yield depth_image
        
    finally:
        pipe.stop()
        
        
def load_detection_model(base_path):
    model = yolov3_model.Model.from_cfg(os.path.join(base_path, "src/core/cfg/yolov3-tiny.cfg"))
    model.tf_model.load_weights(os.path.join(base_path, "logs/20201009-181239/train_ckpts/ckpt_3"))
    return model
        
    
def detect_live(base_path):
    # create live image generator
    live_image_generator = generate_live_images()
    # load detection model
    model = load_detection_model(base_path)
    
    while True:
        # load image
        depth_image = next(live_image_generator)
        depth_image = utils.tf_preprocess_image(depth_image)
        
        # create a batch with a single image
        batch_images = tf.expand_dims(depth_image, axis=0)
        
        tf.print(batch_images.shape)
        
        # predict
        yolo_outputs = model.tf_model.predict(batch_images)
        
        # show result
        conf_threshold = .8
        #utils.draw_grid_detection(batch_images, yolo_outputs, [416, 416, 1], conf_threshold)
        utils.draw_detected_objects(batch_images, yolo_outputs, [416, 416, 1], conf_threshold)
        

def detect_from_file(base_path, file_path):
    # load image
    depth_image = utils.tf_load_preprocessed_image(file_path)
    
    # create a batch with a single image
    batch_images = tf.expand_dims(depth_image, axis=0)
    
    # load detection model
    model = load_detection_model(base_path)
    
    # predict
    yolo_outputs = model.tf_model.predict(batch_images)
    
    # show result
    conf_threshold = .5
    utils.draw_grid_detection(batch_images, yolo_outputs, [416, 416, 1], conf_threshold)
    utils.draw_detected_objects(batch_images, yolo_outputs, [416, 416, 1], conf_threshold)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        base_path = sys.argv[1]
    else:
        base_path = "../"
    
    #detect_from_file(base_path, os.path.join(base_path, 'other/me2.png'))
    detect_live(base_path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    