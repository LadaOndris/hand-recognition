import argparse

from src.system.database.scanner import UsecaseDatabaseScanner
from src.utils.camera import Camera

parser = argparse.ArgumentParser(add_help=False)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='show this help message and exit')

required.add_argument('--dir', type=str, action='store', default=None, required=True,
                      help='The name of the directory that should contain the user-captured gesture database')
required.add_argument('--label', type=str, action='store', default=None, required=True,
                      help='The label of the gesture that is to be captured')
optional.add_argument('--scan-period', type=float, action='store', default=1.0,
                      help='Intervals between each capture in seconds (default: 1)')
optional.add_argument('--camera', type=str, action='store', default='SR305',
                      help='The camera model in use for live capture (default: SR305)')
optional.add_argument('--plot', type=bool, action='store', default=True,
                      help='Plot the captured poses - recommended (default: true)')
args = parser.parse_args()

scanner = UsecaseDatabaseScanner(args.dir, camera=Camera(args.camera), plot_estimation=args.plot)
scanner.scan_into_subdir(args.label, scan_period=args.scan_period)
