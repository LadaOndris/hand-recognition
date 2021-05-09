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


parser = argparse.ArgumentParser(add_help=False)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='show this help message and exit')

required.add_argument('--dir', type=str, action='store', default=None, required=True,
                      help='The name of the directory containg the user-captured gesture database')
required.add_argument('--source', type=str, action='store', required=True,
                      help='The source of images (allowed options: live, dataset)')
required.add_argument('--error-threshold', type=int, action='store', default=120, required=True,
                      help='The pose threshold (JRE threshold)')
optional.add_argument('--orientation-threshold', type=int, action='store', default=90,
                      help='The orientation threshold in angles in the range (maximum: 90, default: 90)')
optional.add_argument('--camera', type=str, action='store', default='SR305',
                      help='The camera model in use for live capture (default: SR305)')
optional.add_argument('--plot', type=bool, action='store', default=True,
                      help='Whether to plot anything.')
optional.add_argument('--plot-feedback', type=bool, action='store', default=True,
                      help='Whether to display the colorbar with JRE errors')
optional.add_argument('--plot-orientation', type=bool, action='store', default=True,
                      help='Whether to display a vector depicting the hand\'s orientation')
args = parser.parse_args()

image_source = get_source_generator(args.source)
live_acceptance = GestureRecognizer(error_thresh=args.error_threshold,
                                    orientation_thresh=args.orientation_threshold,
                                    database_subdir=args.dir,
                                    plot_result=args.plot,
                                    plot_feedback=args.plot_feedback,
                                    plot_orientation=args.plot_orientation,
                                    camera_name=args.camera)
recognizer = live_acceptance.start(image_source)
for result in recognizer:
    print_result(result)
