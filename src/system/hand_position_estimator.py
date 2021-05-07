import argparse

import tensorflow as tf

import src.estimation.configuration as configs
import src.utils.plots as plots
from src.detection.yolov3.cfg.cfg_parser import Model
from src.datasets.generators import get_source_generator
from src.detection.yolov3 import utils
from src.estimation.configuration import Config
from src.estimation.dataset_preprocessing import DatasetPreprocessor
from src.estimation.jgrp2o import JGR_J2O
from src.utils.camera import Camera
from src.utils.config import TEST_YOLO_CONF_THRESHOLD, YOLO_CONFIG_FILE
from src.utils.imaging import tf_resize_image
from src.utils.paths import DOCS_DIR, LOGS_DIR


class HandPositionEstimator:
    """
    Loads a hand detector and hand pose estimator.
    Then, it uses them to estimate the precision position of hands
    either from files, live2 images, or from given images.
    """

    def __init__(self, camera: Camera, config: Config, plot_detection=False, plot_estimation=False, plot_skeleton=True):
        self.camera = camera
        self.plot_detection = plot_detection
        self.plot_estimation = plot_estimation
        self.plot_skeleton = plot_skeleton
        self.resize_mode = 'crop'
        self.detector = self.load_detector(self.resize_mode)
        self.network = JGR_J2O(n_features=196)
        self.estimator = self.load_estimator()
        self.estimation_preprocessor = DatasetPreprocessor(None, self.network.input_size, self.network.out_size,
                                                           camera=self.camera, config=config)
        self.estimation_fig_location = None

    def load_detector(self, resize_mode):
        # Load model based on the preference resize mode
        # because the models were trained with different preprocessing.
        if resize_mode == 'pad':
            weights_file = "20201016-125612/train_ckpts/ckpt_10"  # mode pad
        elif resize_mode == 'crop':
            weights_file = "20210315-143811/train_ckpts/weights.12.h5"  # mode crop
        weights_path = LOGS_DIR.joinpath(weights_file)
        model = Model.from_cfg(YOLO_CONFIG_FILE)
        model.tf_model.load_weights(str(weights_path))
        return model

    def load_estimator(self):
        # Load HPE model and weights

        # weights_path = LOGS_DIR.joinpath('20210329-032745/train_ckpts/weights.04.h5')  # bighand
        # weights_path = LOGS_DIR.joinpath('20210316-035251/train_ckpts/weights.18.h5')  # msra
        # weights_path = LOGS_DIR.joinpath("20210330-024055/train_ckpts/weights.52.h5")  # bighand
        # weights_path = LOGS_DIR.joinpath("20210402-112810/train_ckpts/weights.14.h5")  # bighand from 52.h5
        # weights_path = LOGS_DIR.joinpath("20210403-011340/train_ckpts/weights.15.h5")  # bighand from 52.h5
        # weights_path = LOGS_DIR.joinpath("20210403-212844/train_ckpts/weights.10.h5")  # bighand smaller LR
        # weights_path = LOGS_DIR.joinpath("20210404-133121/train_ckpts/weights.14.h5")  # bighand smaller LR, augmentation, otsus thesh 0.03
        # weights_path = LOGS_DIR.joinpath("20210404-175716/train_ckpts/weights.02.h5")  # bighand smaller LR, augmentation, otsus thesh 0.01
        # weights_path = LOGS_DIR.joinpath("20210405-223012/train_ckpts/weights.20.h5")  # augmentation, LR = e-4
        # weights_path = LOGS_DIR.joinpath("20210407-172950/train_ckpts/weights.18.h5")
        # weights_path = LOGS_DIR.joinpath("20210409-033315/train_ckpts/weights.20.h5")  # batch size 32
        # weights_path = LOGS_DIR.joinpath("20210409-031509/train_ckpts/weights.12.h5")  # batch size 64
        # weights_path = LOGS_DIR.joinpath("20210414-190122/train_ckpts/weights.13.h5")
        # weights_path = LOGS_DIR.joinpath("20210415-233335/train_ckpts/weights.13.h5")
        # weights_path = LOGS_DIR.joinpath("20210417-020242/train_ckpts/weights.13.h5")
        # weights_path = LOGS_DIR.joinpath("20210418-122635/train_ckpts/weights.08.h5")
        # weights_path = LOGS_DIR.joinpath("20210418-200105/train_ckpts/weights.12.h5")
        # weights_path = LOGS_DIR.joinpath("20210423-220702/train_ckpts/weights.13.h5")
        weights_path = LOGS_DIR.joinpath("20210426-125059/train_ckpts/weights.25.h5")  # bighand

        model = self.network.graph()
        model.load_weights(str(weights_path))
        return model

    def estimate_from_file(self, file_path: str, fig_location=None):
        """
        Detect a hand from a single image.
        """
        self.estimation_fig_location = fig_location
        image = self.read_image(file_path)
        return self.estimate_from_image(image)

    def estimate_from_source(self, source_generator, save_folder=None):
        i = 0
        for depth_image in source_generator:
            if save_folder is not None:
                fig_location = save_folder.joinpath(F"{i}.png")
                self.estimation_fig_location = fig_location
            joints = self.estimate_from_image(depth_image)
            yield joints
            i += 1

    def estimate_from_image(self, image):
        """
        Performs hand detection and pose estimation on the given image.

        Parameters
        ----------
        image
            Image pixels should be in milimeters as they are preprocessed
            accordingly for the detector.

        Returns
        -------

        """
        image = tf.convert_to_tensor(image)
        if tf.rank(image) != 3:
            raise Exception("Invalid image rank, expected 3")
        resized_image = self.resize_image_and_depth(image)
        batch_images = tf.expand_dims(resized_image, axis=0)
        boxes = self.detect(batch_images)

        # If detection failed
        if tf.experimental.numpy.allclose(boxes, 0):
            return None
        joints_uvz = self.estimate(batch_images, boxes)
        return joints_uvz

    def detect_from_source(self, source_generator):
        """
        Reads live2 images from RealSense depth camera, and
        detects hands for each frame.
        """
        for depth_image in source_generator:
            depth_image = self.resize_image_and_depth(depth_image)
            batch_images = tf.expand_dims(depth_image, axis=0)
            self.detect(batch_images)

    def detect(self, images):
        """

        Parameters
        ----------
        images

        Returns
        -------
        boxes : shape [batch_size, 4]
            Returns all zeros if non-max suppression did not find any valid boxes.
        """
        detection_batch_images = preprocess_image_for_detection(images)
        yolo_outputs = self.detector.tf_model.predict(detection_batch_images)

        boxes, scores, nums = utils.boxes_from_yolo_outputs(yolo_outputs, self.detector.batch_size,
                                                            self.detector.input_shape,
                                                            TEST_YOLO_CONF_THRESHOLD, iou_thresh=.7, max_boxes=1)
        if self.plot_detection:
            utils.draw_predictions(images[0], boxes[0], nums[0],
                                   fig_location=DOCS_DIR.joinpath('images/deteceted_hand_for_hpe.png'))
        return boxes

    def estimate(self, images, boxes):
        boxes = tf.cast(boxes, dtype=tf.int32)
        normalized_images = self.estimation_preprocessor.preprocess(images, boxes[:, 0, :])
        y_pred = self.estimator.predict(normalized_images)
        uvz_pred = self.estimation_preprocessor.postprocess(y_pred)

        # Plot the inferenced pose
        if self.plot_estimation:
            joints2d = self.estimation_preprocessor.convert_coords_to_local(uvz_pred)
            image = self.estimation_preprocessor.cropped_imgs[0].to_tensor()
            if self.plot_skeleton:
                plots.plot_joints_2d(image, joints2d[0], fig_location=self.estimation_fig_location,
                                     figsize=(4, 4))
            else:
                plots.plot_depth_image(images, fig_location=self.estimation_fig_location)
        return uvz_pred

    def get_cropped_image(self):
        return self.estimation_preprocessor.cropped_imgs[0].to_tensor()

    def convert_to_cropped_coords(self, joints_uvz):
        joints_subregion = self.estimation_preprocessor.convert_coords_to_local(joints_uvz)
        return joints_subregion

    def read_image(self, file_path: str):
        """
        Reads image from file.

        Returns
        -------
        Depth image
        """
        # load image
        depth_image = utils.tf_load_image(file_path, dtype=tf.uint16, shape=self.camera.image_size)
        return depth_image

    def resize_image_and_depth(self, image):
        """
        Resizes to size matching the detector's input shape.
        The unit of image pixels are set as milimeters.
        """
        image = tf_resize_image(image, shape=self.detector.input_shape[:2],
                                resize_mode=self.resize_mode)
        image = set_depth_unit(image, 0.001, self.camera.depth_unit)
        return image


def set_depth_unit(images, target_depth_unit, previous_depth_unit):
    """
    Converts image pixel values to the specified unit.
    """
    dtype = images.dtype
    images = tf.cast(images, dtype=tf.float32)
    images *= previous_depth_unit / target_depth_unit
    images = tf.cast(images, dtype=dtype)
    return images


def preprocess_image_for_detection(images):
    """
    Multiplies pixel values by 8 to match the units expected by the detector.
    Converts image dtype to tf.uint8.

    Parameters
    ----------
    images
        Image pixel values are expected to be in milimeters.
    camera
    """
    dtype = images.dtype
    images = tf.cast(images, dtype=tf.float32)
    images *= 8.00085466544
    images = tf.cast(images, dtype=dtype)
    images = tf.image.convert_image_dtype(images, dtype=tf.uint8)
    return images


def get_estimator(camera):
    camera = Camera(camera)
    config = configs.PredictCustomDataset()
    estimator = HandPositionEstimator(camera, config=config)
    return estimator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, action='store', required=True)
    parser.add_argument('--detect', type=bool, action='store_true', default=False)
    parser.add_argument('--estimate', type=bool, action='store_true', default=False)
    parser.add_argument('--camera', type=str, action='store', default='SR305')
    parser.add_argument('--plot', type=bool, action='store', default=True)
    args = parser.parse_args()

    image_source = get_source_generator(args.source)
    estimator = get_estimator(args.camera)

    if args.detect and args.estimate:
        raise ValueError('Only one of these options can be specified: --detect, --estimate')

    if args.detect:
        estimator.plot_detection = args.plot
        estimator.detect_from_source(image_source)
    if args.estimate:
        estimator.plot_estimation = args.plot
        estimator.estimate_from_source(image_source)
