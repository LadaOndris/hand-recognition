
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datasets.simple_boxes.dataset_bboxes import SimpleBoxesDataset

def draw_bounding_box(axis, bbox):
    if not bbox is None:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1] 
        rect = patches.Rectangle((bbox[0],bbox[1]),w,h,linewidth=1,edgecolor='r',facecolor='none')
        axis.add_patch(rect)
    
def show_images(num):
    dataset = SimpleBoxesDataset(batch_size=num)
    for batch_images, batch_bboxes in dataset.batch_iterator:
        for image, bboxes in zip(batch_images, batch_bboxes):
            fig, ax = plt.subplots(1)
            ax.imshow(np.squeeze(image, axis=-1))
            for bbox in bboxes:
                draw_bounding_box(ax, bbox)
            plt.title('Resized depth image with bounding boxes');
            plt.show()
        break
    
show_images(5)
