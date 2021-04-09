from datasets.handseg150k.dataset import HandsegDataset
from src.detection.rdf.feature_extraction_numba import extract_features, get_pixel_coords, get_feature_offsets
from joblib import dump, load
import timeit
import numpy as np


# to benchmark, always use the same sampled pixels and offsets ->
# generate them beforehand and save

def save_params(pixels, offsets):
    dump(offsets, "offsets_benchmark.joblib")
    dump(pixels, "pixels_benchmark.joblib")


def load_params():
    offsets = load("offsets_benchmark.joblib")
    pixels = load("pixels_benchmark.joblib")
    return pixels, offsets


def create_benchmark_parameters():
    """
    Creates features offsets and pixel coordinates used to
    extract features from the images.
    Then they are saved to files.
    """
    dataset = HandsegDataset()
    depth_images, masks = dataset.load_images(0, 10)

    feature_offsets = get_feature_offsets(2000, depth_images[0].shape)
    pixel_coords = get_pixel_coords(12, depth_images[0].shape)

    save_params(pixel_coords, feature_offsets)


def benchmark(images, reps):
    """
    Benchmarks the feature extraction for given number of images.
    Calls extract_features reps times.

    Returns
    ----------
    Returns the mean execution time.
    """
    dataset = HandsegDataset()
    depth_images, masks = dataset.load_images(0, images)

    execution_times = np.empty(shape=(reps))
    for i in range(reps):
        start = timeit.default_timer()
        _ = extract_features(depth_images, offsets=feature_offsets,
                             pixels=pixel_coords)
        stop = timeit.default_timer()
        execution_times[i] = stop - start
    return np.mean(execution_times)


if __name__ == '__main__':
    pixel_coords, feature_offsets = load_params()
    # feature_offsets = feature_offsets.astype(np.int32)
    """
    feature_offsets = get_feature_offsets(2000, depth_images[0].shape)
    pixel_coords = get_pixel_coords(12, depth_images[0].shape)
    """

    print("Feature offsets", feature_offsets.shape)
    print("Pixel coordinates", pixel_coords.shape)

    mean_exec_time = benchmark(images=100, reps=1)
    print("Feature extraction mean execution time: %.2f" % mean_exec_time)
