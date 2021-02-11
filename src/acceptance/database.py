from src.utils.logs import make_timestamped_dir
from src.utils.paths import CUSTOM_DATASET_DIR


def scan_images(scan_period: float):
    """
    Creates a new directory for the scanned images.
    Scans images in intervals specified by 'scan_period'.

    """
    scan_dir = make_timestamped_dir(CUSTOM_DATASET_DIR)

    pass



if __name__ == '__main__':
    # scan_images(scan_period=1)
    pass
