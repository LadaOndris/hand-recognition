import argparse

from src.system.database.scanner import UsecaseDatabaseScanner
from src.utils.camera import Camera

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, action='store',
                    help='the name of the directory that should contain the user-captured gesture database')
parser.add_argument('label', type=str, action='store',
                    help='the label of the gesture that is to be captured')
parser.add_argument('count', type=int, action='store',
                    help='the number of samples to scan')

parser.add_argument('--scan-period', type=float, action='store', default=1.0,
                    help='intervals between each capture in seconds (default: 1)')
parser.add_argument('--camera', type=str, action='store', default='SR305',
                    help='the camera model in use for live capture (default: SR305)')
parser.add_argument('--hide-plot', action='store_true', default=False,
                    help='hide plots of the captured poses - not recommended')
args = parser.parse_args()

plot = not args.hide_plot
scanner = UsecaseDatabaseScanner(args.directory, camera=Camera(args.camera), plot_estimation=plot)
scanner.scan_into_subdir(args.label, args.count, scan_period=args.scan_period)
