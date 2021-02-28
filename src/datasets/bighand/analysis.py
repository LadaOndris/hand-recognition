
from src.utils.paths import BIGHAND_DATASET_DIR
import numpy as np
from PIL import Image

im = Image.open(BIGHAND_DATASET_DIR.joinpath('Subject_1/151 225/image_D00000111.png'))
image = np.array(im)

pass
