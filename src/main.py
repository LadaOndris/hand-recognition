"""
1. detection
2. pose estimation
3. gesture fulfillment
"""

# from detection.rdf import load_model
import detection
import pose_estimation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import os
from PIL import Image
from src.detection.yolov3 import utils
from src.core.cfg.cfg_parser import Model
from src.utils.paths import LOGS_DIR, SRC_DIR, CUSTOM_DATASET_DIR
from src.utils.config import TEST_YOLO_CONF_THRESHOLD, YOLO_CONFIG_FILE
from src.utils.live import generate_live_images
from src.utils.paths import OTHER_DIR
from src.pose_estimation.preprocessing import ComPreprocessor
from src.pose_estimation.dataset_generator import DatasetGenerator
from src.utils.camera import Camera
import src.utils.plots as plots
from src.pose_estimation.jgr_j2o import JGR_J2O


def load_detection_model():
    model = Model.from_cfg(YOLO_CONFIG_FILE)

    # Also change resize mode down below
    # because the model was trained with different preprocessing mode
    # weights_file = "20201016-125612/train_ckpts/ckpt_10"  # mode pad
    weights_file = "20210315-143811/train_ckpts/weights.12.h5" # mode crop
    weights_path = LOGS_DIR.joinpath(weights_file)
    model.tf_model.load_weights(str(weights_path))
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
        depth_image = utils.tf_resize_image(depth_image)

        # create a batch with a single image
        batch_images = tf.expand_dims(depth_image, axis=0)

        # predict
        yolo_outputs = model.tf_model.predict(batch_images)

        # show result
        # utils.draw_grid_detection(batch_images, yolo_outputs, [416, 416, 1], TEST_YOLO_CONF_THRESHOLD)
        utils.draw_detected_objects(batch_images, yolo_outputs, [416, 416, 1], TEST_YOLO_CONF_THRESHOLD)


def detect_from_file(file_path: str, shape):
    """
    Detect a hand from a single image.
    """
    # ------ Read image ------

    preprocess_image_size = [416, 416, 1]
    # load image
    depth_image = utils.tf_load_image(file_path, dtype=tf.uint16, shape=shape)
    depth_image = utils.tf_resize_image(depth_image, shape=preprocess_image_size[:2], resize_mode='crop')
    # create a batch with a single image
    batch_images = tf.expand_dims(depth_image, axis=0)

    # batch_images *= 8
    detection_batch_images = tf.image.convert_image_dtype(batch_images, dtype=tf.uint8)

    # ------ Pass to detection stage ------

    # load detection model
    model = load_detection_model()

    # predict
    yolo_outputs = model.tf_model.predict(detection_batch_images)

    # show result
    # utils.draw_grid_detection(batch_images, yolo_outputs, preprocess_image_size, TEST_YOLO_CONF_THRESHOLD)
    # utils.draw_detected_objects(batch_images, yolo_outputs, preprocess_image_size, TEST_YOLO_CONF_THRESHOLD,
    #                            max_boxes=1)

    boxes, scores, nums = utils.boxes_from_yolo_outputs(yolo_outputs, model.batch_size, preprocess_image_size,
                                                        0.5, iou_thresh=.7, max_boxes=1)

    # utils.draw_predictions(detection_batch_images[0], boxes[0], nums[0], fig_location=None)
    # Convert values to milimeters - what the actual fuck???
    # The custom SR305 camera returns pixels in 0.12498664727900177 mm per value
    # To correct the values and to be in mm, divide by 8.00085466544
    # pose_images = tf.cast(batch_images, dtype=tf.float32) / 8.00085466544
    pose_images = batch_images
    utils.draw_predictions(batch_images[0], boxes[0], nums[0], fig_location=None)

    boxes = tf.cast(boxes, dtype=tf.int32)
    # utils.draw_predictions(pose_images[0], boxes[0, tf.newaxis, :], nums[0], fig_location=None)

    # ------ Pass to hand pose estimation stage ------

    # Load HPE model and weights
    weights_path = LOGS_DIR.joinpath('20210316-035251/train_ckpts/weights.18.h5')
    # weights_path = LOGS_DIR.joinpath("20210323-160416/train_ckpts/weights.10.h5")
    network = JGR_J2O()
    model = network.graph()
    model.load_weights(str(weights_path))

    # Preprocess the image
    camera = Camera('custom')
    dataset_generator = DatasetGenerator(None, preprocess_image_size, network.input_size, network.out_size,
                                         camera=camera, augment=False)
    normalized_images = dataset_generator.preprocess(pose_images, boxes[:, 0, :], cube_size=[150, 150, 150])

    # Inference
    y_pred = model.predict(normalized_images)

    # Postprocess inferenced pose
    uvz_pred = dataset_generator.postprocess(y_pred)
    joints2d = uvz_pred[..., :2] - dataset_generator.bboxes[..., tf.newaxis, :2]

    # Plot the inferenced pose
    plots.plot_joints_2d(normalized_images[0], joints2d[0])

    pass


if __name__ == '__main__':
    # detect_from_file(str(CUSTOM_DATASET_DIR.joinpath('20210326-153934/35.png')), [720, 1280])
    detect_from_file(str(CUSTOM_DATASET_DIR.joinpath('20210326-230610/200.png')), [480, 640])
    # detect_from_file(str(OTHER_DIR.joinpath('me.png')))
    # detect_live()
    pass
