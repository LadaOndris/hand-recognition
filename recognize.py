import argparse

from src.acceptance.gesture_acceptance_result import GestureAcceptanceResult
from src.datasets.generators import get_source_generator
from src.system.gesture_recognizer import GestureRecognizer


def print_result(result: GestureAcceptanceResult):
    if not result.is_gesture_valid:
        result.gesture_label = 'None'

    print(F"Gesture: {result.gesture_label}\t"
          F"JRE: {result.gesture_jre}\t"
          F"Orient. diff: {result.angle_difference:.2f}")


parser = argparse.ArgumentParser()
parser.add_argument('source', type=str, action='store',
                    help='the source of images (allowed options: live, dataset)')
parser.add_argument('directory', type=str, action='store',
                    help='the name of the directory containg the user-captured gesture database')

parser.add_argument('--error-threshold', type=int, action='store', default=120,
                    help='the pose (JRE) threshold (default: 120)')
parser.add_argument('--orientation-threshold', type=int, action='store', default=90,
                    help='the orientation threshold in angles (maximum: 90, default: 90)')
parser.add_argument('--camera', type=str, action='store', default='SR305',
                    help='the camera model in use for live capture (default: SR305)')
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot the result of gesture recognition')
parser.add_argument('--hide-feedback', action='store_true', default=False,
                    help='hide the colorbar with JRE errors')
parser.add_argument('--hide-orientation', action='store_true', default=False,
                    help='hide the vector depicting the hand\'s orientation')
args = parser.parse_args()

plot_feedback = not args.hide_feedback
plot_orientation = not args.hide_orientation

image_source = get_source_generator(args.source)
live_acceptance = GestureRecognizer(error_thresh=args.error_threshold,
                                    orientation_thresh=args.orientation_threshold,
                                    database_subdir=args.directory,
                                    plot_result=args.plot,
                                    plot_feedback=plot_feedback,
                                    plot_orientation=plot_orientation,
                                    camera_name=args.camera)
recognizer = live_acceptance.start(image_source)
for result in recognizer:
    print_result(result)
