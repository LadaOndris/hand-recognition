import argparse

from src.acceptance.predict import GestureAcceptanceResult
from src.datasets.generators import get_source_generator
from src.system.gesture_recognizer import GestureRecognizer


def print_result(result: GestureAcceptanceResult):
    if not result.is_gesture_valid:
        result.gesture_label = 'None'

    print(F"Gesture: {result.gesture_label}, "
          F"JRE: {result.gesture_jre}, "
          F"Orient. diff: {result.angle_difference}")


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, action='store', default=None, required=True)
parser.add_argument('--source', type=str, action='store', required=True)
parser.add_argument('--error-threshold', type=int, action='store', default=120, required=True)
parser.add_argument('--orientation-threshold', type=int, action='store', default=90)
parser.add_argument('--camera', type=str, action='store', default='SR305')
parser.add_argument('--plot', type=bool, action='store', default=True)
parser.add_argument('--plot-feedback', type=bool, action='store', default=True)
args = parser.parse_args()

image_source = get_source_generator(args.source)
live_acceptance = GestureRecognizer(error_thresh=args.error_threshold,
                                    orientation_thresh=args.orientation_threshold,
                                    database_subdir=args.dir,
                                    plot_feedback=args.plot_feedback,
                                    plot_result=args.plot,
                                    camera_name=args.camera)
recognizer = live_acceptance.start(image_source)
for result in recognizer:
    print_result(result)
