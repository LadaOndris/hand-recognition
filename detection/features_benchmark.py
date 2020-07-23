from datasets.handseg150k.dataset import HandsegDataset, HUGE_INT
from feature_extraction_numba import extract_features, get_pixel_coords, get_feature_offsets
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
    dataset = HandsegDataset()
    depth_images, masks = dataset.load_images(0, 10)
    
    feature_offsets = get_feature_offsets(2000, depth_images[0].shape)
    pixel_coords = get_pixel_coords(12, depth_images[0].shape)
    
    save_params(pixel_coords, feature_offsets)
    
#pixel_coords, feature_offsets = load_params()
#feature_offsets = feature_offsets.astype(np.int32)

dataset = HandsegDataset()
depth_images, masks = dataset.load_images(0, 1)

feature_offsets = get_feature_offsets(1000, depth_images[0].shape)
pixel_coords = get_pixel_coords(24, depth_images[0].shape)

print(feature_offsets.shape)
print(pixel_coords.shape)



start = timeit.default_timer()
features = extract_features(depth_images, offsets = feature_offsets, 
                            pixels = pixel_coords)
stop = timeit.default_timer()
execution_time = stop - start
print("Feature extraction execution time: %.2f" % execution_time)

print(features.shape, features.dtype)