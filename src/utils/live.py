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


def intrinsic_parameters():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480)
    profile = pipe.start(cfg)
    stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intrinsics = stream.get_intrinsics()
    """
    fx = fy = 476.0068054199219
    ppx, ppy = 313.6830139, 242.7547302
    """
    pass


if __name__ == '__main__':
    intrinsic_parameters()
