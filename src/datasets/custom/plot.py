"""
This file is used to plot samples from the custom dataset
"""

import src.utils.plots as plots
from src.utils.paths import CUSTOM_DATASET_DIR, DOCS_DIR
from PIL import Image
import numpy as np


def plot_image(filepath, fig_location=None):
    path = CUSTOM_DATASET_DIR.joinpath(filepath)
    img = np.asarray(Image.open(path))
    plots.plot_depth_image(img, figsize=(4, 3), fig_location=fig_location)


save_fig_pattern = DOCS_DIR.joinpath('figures/datasets/CustomSampleImage{}.png')
save_fig_pattern = str(save_fig_pattern)
plot_image('20210326-230536/110.png', save_fig_pattern.format(1))
plot_image('20210326-232147/70.png', save_fig_pattern.format(2))
plot_image('20210326-233307/20.png', save_fig_pattern.format(3))
plot_image('20210326-232751/850.png', save_fig_pattern.format(4))
