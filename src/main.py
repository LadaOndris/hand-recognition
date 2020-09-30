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
from detection.rdf import predict_boundary
from PIL import Image 

def detect_and_estimate(depth_im):
    detection_model = detection.rdf.load_model('', '')
    bounding_box = detection.predict_boundary(detection_model, depth_im)
    pose_model = pose_estimation.deep_prior.load_model('', '')
    pose = pose_estimation.deep_prior.predict_pose(pose_model, bounding_box)
    
    
def get_live_depth_im():
    
    return
    
def print_live_images():
        
    model, pixels, offsets  = detection.rdf.load_model(
        './detection/saved_models/incr_trees_per_iteration', '_5')
        
    pipe = rs.pipeline()
    profile = pipe.start()
    try:
      shot = 0
      while True:
        frameset = pipe.wait_for_frames()
        depth_frame = frameset.get_depth_frame()
        depth_image = np.array(depth_frame.get_data())
        
        #detection.rdf.predict_boundary(model, pixels, offsets, depth_image)
        
        plt.imshow(depth_image)
        plt.show()
        if shot == 20:
            im = Image.fromarray(depth_image)
            im.save("me2.png")
            break
        shot += 1
    finally:
        pipe.stop()
        
    
def live_detect():
    model_path = './detection/saved_models/incr_trees_per_iteration'
    model_namesuffix = '_5'
    
    detection_model = detection.rdf.load_model(model_path, model_namesuffix)
    depth_img = get_live_depth_im()
    
    
print_live_images()