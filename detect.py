import argparse

import src.estimation.configuration as configs
from src.datasets.generators import get_source_generator
from src.system.hand_position_estimator import HandPositionEstimator
from src.utils.camera import Camera


def get_estimator(camera: str) -> HandPositionEstimator:
    camera = Camera(camera)
    config = configs.PredictCustomDataset()
    estimator = HandPositionEstimator(camera, config=config)
    return estimator


parser = argparse.ArgumentParser()
parser.add_argument('source', type=str, action='store',
                    help='The source of images (allowed options: live, dataset)')
parser.add_argument('--camera', type=str, action='store', default='SR305',
                    help='The camera model in use for live capture (default: SR305)')
parser.add_argument('--plot', type=bool, action='store', default=True,
                    help='Whether to plot the result of detection (default: true)')
parser.add_argument('--num-detections', action='store', type=int, default=1,
                    help='The maximum number of bounding boxes for hand detection (default: 1)')
args = parser.parse_args()

image_source = get_source_generator(args.source)
estimator = get_estimator(args.camera)
estimator.plot_detection = args.plot
detect_generator = estimator.detect_from_source(image_source, args.num_detections)
for boxes in detect_generator:
    print('Bounding boxes:', boxes.numpy())
