from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from pathlib import Path
import numpy as np
import os
from datasets.handseg150k.dataset import HandsegDataset

dirname = os.path.dirname(__file__)
    

def foo():
    depth_im = np.array(Image.open(os.path.join(dirname, 'images/user-4.00000003.png')))# Loading depth image
    mask_im = np.array(Image.open(os.path.join(dirname, 'masks/user-4.00000003.png')))#  Loading mask image
    depth_im = depth_im.astype(np.float32)# Converting to float
    mean = np.mean(depth_im)
    print(Counter(depth_im.flatten()))
    #plt.hist(mask_im, bins=100)
    #plt.show()
    mean_depth_ims = 10000.0 # Mean value of the depth images
    depth_im /= mean_depth_ims # Normalizing depth image
    plt.imshow(depth_im); plt.title('Depth Image'); plt.show() # Displaying Depth Image
    plt.imshow(mask_im); plt.title('Mask Image'); plt.show() # Displaying Mask Image
    
    
    print(depth_im.shape)
    
def show_images(num):
    dataset = HandsegDataset()
    images, masks = dataset.load_images(start_index = 0, end_index = num - 1)
    for i, m in zip(images, masks):
        #print(np.histogram(m, bins=[0,1,2,3]))
        plt.imshow(i); plt.title('Depth Image'); plt.show() # Displaying Depth Image
        plt.imshow(m); plt.title('Mask Image'); plt.show() # Displaying Mask Image

def get_bbox_from_mask(mask, hand_label):
    hand_coords = np.where(mask == hand_label)
    if (len(hand_coords[0]) == 0):
        return None
    
    x_coords = hand_coords[1]
    y_coords = hand_coords[0]
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    
    return x_min, y_min, x_max, y_max

def draw_bounding_box(axis, bbox):
    if not bbox is None:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1] 
        rect = patches.Rectangle((bbox[0],bbox[1]),w,h,linewidth=1,edgecolor='r',facecolor='none')
        axis.add_patch(rect)
    

def generate_bounding_boxes(print_images = False, save_to_file = False, bboxes_filename=''):
    filenames = os.listdir('masks/')
    
    with open(bboxes_filename, 'w') as bboxes_file:
        for filename in filenames:
            image = np.array(Image.open(os.path.join(dirname, 'images', filename)))
            mask = np.array(Image.open(os.path.join(dirname, 'masks', filename)))
            #print(mask.shape)
            #print(np.histogram(mask, bins=[0,1,2,3]))
            
            first_bbox = get_bbox_from_mask(mask, 1)
            second_bbox = get_bbox_from_mask(mask, 2)
            
            if save_to_file:
                bboxes_file.write(filename)
                if not first_bbox is None:
                    bboxes_file.write(F" {first_bbox[0]} {first_bbox[1]} {first_bbox[2]} {first_bbox[3]};")
                if not second_bbox is None:
                    bboxes_file.write(F" {second_bbox[0]} {second_bbox[1]} {second_bbox[2]} {second_bbox[3]};")
                bboxes_file.write('\n')
            
            #print(first_bbox, second_bbox)
            
            if print_images:
                fig, ax = plt.subplots(1)
                ax.imshow(image)
                draw_bounding_box(ax, first_bbox)
                draw_bounding_box(ax, second_bbox)
                plt.title('Depth image with bounding boxes');
                plt.show()
    
generate_bounding_boxes(print_images=False, save_to_file=True, bboxes_filename='bounding_boxes.txt')