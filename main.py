"""
1. detection
2. pose estimation
3. gesture fulfillment
"""

#from detection.rdf import load_model
import detection
import pose_estimation
import matplotlib.pyplot as plt  

def detect_and_estimate(depth_im):
    detection_model = detection.rdf.load_model('', '')
    bounding_box = detection.rdf.predict_boundary(detection_model, depth_im)
    pose_model = pose_estimation.deep_prior.load_model('', '')
    pose = pose_estimation.deep_prior.predict_pose(pose_model, bounding_box)
    
    
    