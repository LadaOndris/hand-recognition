from src.utils.paths import CUSTOM_DATASET_DIR
from src.utils.logs import make_timestamped_dir
from src.utils.live import generate_live_images
from src.utils.plots import _plot_depth_image, plot_depth_image
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


def save_16bit_png(raw_image, path):
    raw_image = raw_image.astype(np.uint16)
    array_buffer = raw_image.tobytes()
    img = Image.new("I", raw_image.T.shape)
    img.frombytes(array_buffer, 'raw', "I;16")
    img.save(path)


def clean_max_depth(image_data, max_depth_mm=2000):
    image_data[image_data > max_depth_mm] = 0
    return image_data


def start_live_capture(plot=False):
    generator = generate_live_images()
    dir = make_timestamped_dir(CUSTOM_DATASET_DIR)
    print("Capture has starated...")
    for i, image_array in enumerate(generator):
        # image_array = clean_max_depth(image_array, 3000)
        if plot:
            plot_depth_image(image_array)
        path = os.path.join(dir, F"{i}.png")
        save_16bit_png(np.squeeze(image_array), path)
        print(path)
        plt.pause(0.05)


def show_captured(dirname):
    dirpath = CUSTOM_DATASET_DIR.joinpath(dirname)
    dirs_and_files = dirpath.iterdir()
    files = [file for file in dirs_and_files if file.is_file()]
    files.sort(key=os.path.getmtime)
    plt.ion()
    fig, ax = plt.subplots()

    for file in files:
        print(file)
        image = np.array(Image.open(file))
        _plot_depth_image(ax, image)
        plt.draw()
        plt.pause(0.005)
        plt.cla()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    start_live_capture()
    # show_captured('20210326-233053')

