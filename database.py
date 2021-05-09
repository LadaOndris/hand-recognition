import argparse

from src.system.database.scanner import UsecaseDatabaseScanner
from src.utils.camera import Camera

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, action='store', default=None, required=True)
parser.add_argument('--label', type=str, action='store', default=None, required=True)
parser.add_argument('--scan-period', type=float, action='store', default=1.0)
parser.add_argument('--camera', type=str, action='store', default='SR305')
parser.add_argument('--plot', type=bool, action='store', default=True)
args = parser.parse_args()

scanner = UsecaseDatabaseScanner(args.dir, camera=Camera(args.camera), plot_estimation=args.plot)
scanner.scan_into_subdir(args.label, scan_period=args.scan_period)
