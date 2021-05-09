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


parser = argparse.ArgumentParser(add_help=False)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='show this help message and exit')

required.add_argument('--source', type=str, action='store', required=True,
                      help='The source of images (allowed options: live, dataset)')
optional.add_argument('--camera', type=str, action='store', default='SR305',
                      help='The camera model in use for live capture (default: SR305)')
optional.add_argument('--plot', type=bool, action='store', default=True,
                      help='Whether to plot the result of detection (default: true)')
optional.add_argument('--num-detections', action='store', type=int, default=1,
                      help='The maximum number of bounding boxes for hand detection (default: 1)')
args = parser.parse_args()

image_source = get_source_generator(args.source)
estimator = get_estimator(args.camera)
estimator.plot_detection = args.plot
detect_generator = estimator.detect_from_source(image_source, args.num_detections)
for boxes in detect_generator:
    print('Bounding boxes:', boxes.numpy())
