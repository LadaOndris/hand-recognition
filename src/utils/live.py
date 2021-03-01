import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt


def generate_live_images():
    pipe = rs.pipeline()
    profile = pipe.start()
    try:
        while True:
            frameset = pipe.wait_for_frames()
            depth_frame = frameset.get_depth_frame()
            depth_image = np.array(depth_frame.get_data())
            depth_image = depth_image[..., np.newaxis]

            yield depth_image

    finally:
        pipe.stop()


def print_live_images(num=None):
    generator = generate_live_images()

    i = 0
    while True:
        if i == num:
            break
        i += 1

        depth_image = next(generator)
        plt.imshow(depth_image)
        plt.show()
