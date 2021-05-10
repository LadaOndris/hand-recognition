import argparse

import src.estimation.configuration as configs
from src.datasets.generators import get_source_generator
from src.system.hand_position_estimator import HandPositionEstimator
from src.utils.camera import Camera


def get_estimator(camera: str, plot: bool) -> HandPositionEstimator:
    camera = Camera(camera)
    config = configs.PredictCustomDataset()
    estimator = HandPositionEstimator(camera, config=config, plot_estimation=plot)
    return estimator


parser = argparse.ArgumentParser()
parser.add_argument('source', type=str, action='store',
                    help='The source of images (allowed options: live, dataset)')
parser.add_argument('--camera', type=str, action='store', default='SR305',
                    help='The camera model in use for live capture (default: SR305)')
parser.add_argument('--plot', type=bool, action='store', default=True,
                    help='Whether to plot the estimation (default: true)')
args = parser.parse_args()

image_source = get_source_generator(args.source)
estimator = get_estimator(args.camera, args.plot)
estimation_generator = estimator.estimate_from_source(image_source)
for joints in estimation_generator:
    print('Joints\' coordinates:\n', joints.numpy())
