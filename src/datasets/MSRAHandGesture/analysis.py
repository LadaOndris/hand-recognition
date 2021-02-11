import pathlib
import os
import numpy as np
import struct
from src.utils.paths import MSRAHANDGESTURE_DATASET_DIR
from src.utils.plots import plot_depth_image

file = MSRAHANDGESTURE_DATASET_DIR.joinpath('P0/1/000100_depth.bin')
with open(file, 'rb') as f:
    total_width, total_height = struct.unpack('ii', f.read(4 * 2))
    left, top, right, bottom = struct.unpack('i' * 4, f.read(4 * 4))
    width = right - left
    height = bottom - top

    image_data = f.read()
    values = len(image_data) // 4
    image = struct.unpack(F"{values}f", image_data)
    image = np.array(image)
    image = image.reshape([height, width])
    plot_depth_image(image)
    pass
