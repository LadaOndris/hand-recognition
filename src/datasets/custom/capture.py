from src.utils.paths import CUSTOM_DATASET_DIR
from src.utils.logs import make_timestamped_dir
from src.utils.live import generate_live_images
from src.utils.plots import plot_depth_image
from PIL import Image
import numpy as np
import os


def save_16bit_png(raw_image, path):
    raw_image = raw_image.astype(np.uint16)
    array_buffer = raw_image.tobytes()
    img = Image.new("I", raw_image.T.shape)
    img.frombytes(array_buffer, 'raw', "I;16")
    img.save(path)


def clean_max_depth(image_data, max_depth_mm=2000):
    image_data[image_data > max_depth_mm] = 0
    return image_data


generator = generate_live_images()
dir = make_timestamped_dir(CUSTOM_DATASET_DIR)

for i, image_array in enumerate(generator):
    image_array = clean_max_depth(image_array)
    plot_depth_image(image_array)
    path = os.path.join(dir, F"{i}.png")
    save_16bit_png(np.squeeze(image_array), path)
