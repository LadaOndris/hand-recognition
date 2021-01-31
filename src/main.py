"""
1. detection
2. pose estimation
3. gesture fulfillment
"""

# from detection.rdf import load_model
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
from src.utils.paths import LOGS_DIR, SRC_DIR
from src.utils.config import TEST_YOLO_CONF_THRESHOLD


def print_live_images(num=None):
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


def load_detection_model():
    model = yolov3_model.Model.from_cfg(SRC_DIR.joinpath("core/cfg/yolov3-tiny.cfg"))
    model.tf_model.load_weights(LOGS_DIR.joinpath("20201016-125612/train_ckpts/ckpt_10"))
    return model


def detect_live():
    """
    Reads live images from RealSense depth camera, and
    detects hands for each frame.
    """
    # create live image generator
    live_image_generator = generate_live_images()
    # load detection model
    model = load_detection_model()

    while True:
        # load image
        depth_image = next(live_image_generator)
        depth_image = utils.tf_preprocess_image(depth_image)

        # create a batch with a single image
        batch_images = tf.expand_dims(depth_image, axis=0)

        # predict
        yolo_outputs = model.tf_model.predict(batch_images)

        # show result
        # utils.draw_grid_detection(batch_images, yolo_outputs, [416, 416, 1], TEST_YOLO_CONF_THRESHOLD)
        utils.draw_detected_objects(batch_images, yolo_outputs, [416, 416, 1], TEST_YOLO_CONF_THRESHOLD)


def detect_from_file(file_path):
    """
    Detect a hand from a single image.
    """
    # load image
    depth_image = utils.tf_load_preprocessed_image(file_path)

    # create a batch with a single image
    batch_images = tf.expand_dims(depth_image, axis=0)

    # load detection model
    model = load_detection_model()

    # predict
    yolo_outputs = model.tf_model.predict(batch_images)

    # show result
    utils.draw_grid_detection(batch_images, yolo_outputs, [416, 416, 1], TEST_YOLO_CONF_THRESHOLD)
    utils.draw_detected_objects(batch_images, yolo_outputs, [416, 416, 1], TEST_YOLO_CONF_THRESHOLD)


if __name__ == '__main__':
    # detect_from_file(base_path, os.path.join(base_path, 'other/me2.png'))
    detect_live()
