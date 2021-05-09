import argparse

from src.system.database.scanner import UsecaseDatabaseScanner
from src.utils.camera import Camera

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, action='store',
                    help='The name of the directory that should contain the user-captured gesture database')
parser.add_argument('label', type=str, action='store',
                    help='The label of the gesture that is to be captured')
parser.add_argument('count', type=int, action='store',
                    help='The number of samples to scan')

parser.add_argument('--scan-period', type=float, action='store', default=1.0,
                    help='Intervals between each capture in seconds (default: 1)')
parser.add_argument('--camera', type=str, action='store', default='SR305',
                    help='The camera model in use for live capture (default: SR305)')
parser.add_argument('--plot', type=bool, action='store', default=True,
                    help='Plot the captured poses - recommended (default: true)')
args = parser.parse_args()

scanner = UsecaseDatabaseScanner(args.directory, camera=Camera(args.camera), plot_estimation=args.plot)
scanner.scan_into_subdir(args.label, args.count, scan_period=args.scan_period)
