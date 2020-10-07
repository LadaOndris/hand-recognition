"""
Created on Tue Jul  7 21:11:25 2020

@author: Ladislav Ondris
"""

import numpy as np
import os
import cv2

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from joblib import dump, load
from datetime import datetime
from matplotlib import pyplot as plt

from src.datasets.handseg150k.dataset import HandsegDataset, HUGE_INT
from src.detection.rdf.feature_extraction_numba import extract_features, extract_features_and_labels, \
    get_pixel_coords, get_feature_offsets
    
dataset = HandsegDataset()

def get_current_timestamp():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    return dt_string

def load_dataset(start_index, end_index, test_size = 0.2):
    features_offsets = get_feature_offsets(500, dataset.image_shape)
    pixels_coords = get_pixel_coords(18, dataset.image_shape)
    depth_images, masks = dataset.load_images(start_index, end_index)
    features, labels = extract_features_and_labels(depth_images, masks, offsets=features_offsets, pixels=pixels_coords)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test_size)
    return (X_train, X_test, y_train, y_test), pixels_coords, features_offsets

def train_model(X_train, y_train):
    rdf = RandomForestClassifier(max_depth=20, bootstrap=True, n_estimators=64, n_jobs=-1)
    rdf.fit(X_train, y_train)
    return rdf

def save_model(model, pixels, offsets, path, timestamp = True, name_suffix = ""):
    if timestamp == True:
        timestamp = get_current_timestamp()
    else:
        timestamp = ""
    dump(model, os.path.join(path, F"rdf_{timestamp}{name_suffix}.joblib")) 
    dump(offsets, os.path.join(path, F"offsets_{timestamp}{name_suffix}.joblib"))
    dump(pixels, os.path.join(path, F"pixels_{timestamp}{name_suffix}.joblib"))
    
def load_model(path, suffix):
    features_offsets = load(os.path.join(path, F"offsets_{suffix}.joblib"))
    pixels_coords = load(os.path.join(path, F"pixels_{suffix}.joblib"))
    pixels_coords = np.array(pixels_coords)
    model = load(os.path.join(path, F"rdf_{suffix}.joblib"))
    return model, pixels_coords, features_offsets

def evaluate(model, X_test, y_test):
    # determine the accuracy predicted by the model
    threshold = 0.35
    predicted_proba = model.predict_proba(X_test)
    predicted = (predicted_proba [:,1] >= threshold).astype('int')
    
    positive_pixels = sum([l for l in y_test if l == 1])
    print("Total positive:", positive_pixels, "out of", len(y_test), "pixels")
    print("Precision:", precision_score(y_test, predicted))
    print("Recall:", recall_score(y_test, predicted))
    print("F1 score:", f1_score(y_test, predicted)) 
    
def create_train_save():
    print("Loading dataset", flush=True)
    data, pixels, offsets = load_dataset(0, 300, test_size=0.1)
    X_train, X_test, y_train, y_test = data
    print("Training model", flush=True)
    model = train_model(X_train, y_train)
    print("Saving model", flush=True)
    save_model(model, pixels, offsets, './saved_models')
    print("Evaluating model", flush=True)
    evaluate(model, X_test, y_test)
    
    
def create_train_incrementally_save(features_per_pixel, 
                                    trees_per_iteration = 1,
                                    forest_max_depth = 20,
                                    name_suffix = "", 
                                    save_path = "./saved_models",
                                    pixels_coords = None,
                                    features_offsets = None,
                                    first_image_index = 0,
                                    last_image_index = 5000):
    rdf = RandomForestClassifier(max_depth=forest_max_depth, bootstrap=True, n_estimators=1, 
                                 n_jobs=-1, warm_start = True)
    
    if features_offsets is None:
        features_offsets = get_feature_offsets(features_per_pixel, dataset.image_shape)
    if pixels_coords is None:
        pixels_coords = get_pixel_coords(18, dataset.image_shape)
    
    step = 50 * trees_per_iteration
    for i in range(first_image_index, last_image_index, step):
        print(F"Loading dataset {i}", flush=True)
        depth_images, masks = dataset.load_images(i, i + step)
        X, y = extract_features_and_labels(depth_images, masks, 
                                           pixels = pixels_coords,
                                           offsets = features_offsets)
        rdf.n_estimators += trees_per_iteration
        rdf.fit(X, y)
        
    print("Saving model", flush=True)
    save_model(rdf, pixels_coords, features_offsets, save_path,
               timestamp=False, name_suffix=name_suffix)

"""
Good models: 20200712_175549, 20200712_231245
"""
def load_evaluate():
    print("Loading model", flush=True)
    model, pixels, offsets = load_model('/home/lada/projects/IBT/detection', '20200713_105103')
    model, pixels, offsets  = load_model(
        '/home/lada/projects/IBT/detection/saved_models/incr_trees_per_iteration', 
        '_5')
    print("Loading dataset", flush=True)
    depth_images, masks = dataset.load_images(5000, 5050)
    X_test, y_test = extract_features_and_labels(depth_images, masks, offsets=offsets, pixels=pixels)
    print("Evaluating model", flush=True)
    evaluate(model, X_test, y_test)


def predict_all_pixels():
    model, pixels, offsets  = load_model(
        '/home/lada/projects/IBT/detection/saved_models', 
        '_50k')
    depth_image, mask = dataset.load_image(0)
    depth_image = depth_image.astype(np.float32)
    X, y = extract_features_and_labels(depth_image[np.newaxis,:], mask[np.newaxis,:], offsets=offsets, pixels=pixels)
    predicted_proba = model.predict_proba(X)
    predicted = (predicted_proba[:,1] >= 0.5).astype('int')
    print("Precision:", precision_score(y, predicted))
    hand = [pix for pred, pix in zip(predicted, pixels) if pred == 1]
    hand = np.array(hand)
    
    # normalize image, convert to rgb
    depth_image -= depth_image.min()
    depth_image[depth_image > 10000.0] = 0
    depth_image *= 1.0/depth_image.max()
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
    depth_image = 1 - depth_image 
    
    # draw predicted hand pixels on the depth map
    plt.imshow(depth_image, vmin=0, vmax=255);
    if len(hand) > 0:
        plt.scatter(hand[:,1], hand[:,0], c='r', s=3)
    plt.title('Depth Image'); 
    plt.show()
    #plt.imshow(mask, cmap='BuPu'); plt.title('Mask Image'); plt.show()
    

def evaluate_image():
    model, pixels, offsets  = load_model('/home/lada/projects/IBT/detection', '20200712_175549')
    depth_image, mask = dataset.load_image(1)
    X, y = extract_features_and_labels(depth_image, mask, offsets=offsets, pixels=pixels)
    evaluate(model, X, y)
    
def create_rdf_for_different_number_of_features():
    # max number of features 2000
    # further it's computionally intensive
    # let's incrementally train in 50 image batches
    # features number 50 - 2000
    for features_per_pixel in range(50, 2001, 50):
        create_train_incrementally_save(features_per_pixel, 
                                        name_suffix = F"_{features_per_pixel}",
                                        save_path="./saved_models/features_count_test")

def evaluate_features_count_test():
    precision = []
    recall = []
    f1score = []
    
    for features_per_pixel in range(50, 2001, 50):
        print(F"\nEvaluation of {features_per_pixel} model")
        rdf, pixels, offsets = load_model('./saved_models/features_count_test', F"_{features_per_pixel}")
        y_pred = []
        y = []
        
        step = 50
        for i in range(5000, 10000, step):
            depth_images, masks = dataset.load_images(i, i + step)
            X_test, y_test = extract_features_and_labels(depth_images, masks, offsets=offsets, pixels=pixels)
            
            predicted_proba = rdf.predict_proba(X_test)
            predicted_subset = (predicted_proba [:,1] >= 0.5).astype('int')
            y_pred.extend(predicted_subset)
            y.extend(y_test)
            
        precision.append(precision_score(y, y_pred))
        recall.append(recall_score(y, y_pred))
        f1score.append(f1_score(y, y_pred))
        
        print("Precision:", precision[-1])
        print("Recall:", recall[-1])
        print("F1 score:", f1score[-1]) 
    
    print(precision)
    print(recall)
    print(f1score)
   
def create_rdf_for_different_number_of_trees_per_iteration():
    for i in range(1, 25):
        create_train_incrementally_save(
            500, trees_per_iteration = i,
            name_suffix = F"_{i}",
            save_path="./saved_models/incr_trees_per_iteration")
        
def evaluate_number_of_trees_per_iter():
    precision = []
    recall = []
    f1score = []
    
    for trees_per_iter in range(1, 13):
        print(F"\nEvaluation of {trees_per_iter} model")
        rdf, pixels, offsets = load_model('./saved_models/incr_trees_per_iteration', F"_{trees_per_iter}")
        y_pred = []
        y = []
        
        step = 50
        for i in range(150000, 155000, step):
            depth_images, masks = dataset.load_images(i, i + step)
            X_test, y_test = extract_features_and_labels(depth_images, masks, offsets=offsets, pixels=pixels)
            
            predicted_proba = rdf.predict_proba(X_test)
            predicted_subset = (predicted_proba [:,1] >= 0.5).astype('int')
            y_pred.extend(predicted_subset)
            y.extend(y_test)
            
        precision.append(precision_score(y, y_pred))
        recall.append(recall_score(y, y_pred))
        f1score.append(f1_score(y, y_pred))
        
        print("Precision:", precision[-1])
        print("Recall:", recall[-1])
        print("F1 score:", f1score[-1]) 
    
    print(precision)
    print(recall)
    print(f1score)


def create_rdf_with_different_tree_height():
    
    # make sure the parameters stay the same, especially features
    features_per_pixel = 500
    features_offsets = get_feature_offsets(features_per_pixel, dataset.image_shape)
    pixels_coords = get_pixel_coords(18, dataset.image_shape)
    
    for i in range(4, 25, 2):
        create_train_incrementally_save(
            features_per_pixel, 
            forest_max_depth=i,
            trees_per_iteration = 5,
            name_suffix = F"_{i}",
            save_path="./saved_models/tree_height_test",
            pixels_coords = pixels_coords,
            features_offsets = features_offsets)
    
 
def evaluate_tree_height():
    precision = []
    recall = []
    f1score = []
    
    for tree_height in range(4, 25, 2):
        print(F"\nEvaluation of {tree_height} model")
        rdf, pixels, offsets = load_model('./saved_models/tree_height_test', F"_{tree_height}")
        y_pred = []
        y = []
        
        step = 200
        for i in range(150000, 155000, step):
            depth_images, masks = dataset.load_images(i, i + step)
            X_test, y_test = extract_features_and_labels(depth_images, masks, offsets=offsets, pixels=pixels)
            
            predicted_proba = rdf.predict_proba(X_test)
            predicted_subset = (predicted_proba [:,1] >= 0.5).astype('int')
            y_pred.extend(predicted_subset)
            y.extend(y_test)
            
        precision.append(precision_score(y, y_pred))
        recall.append(recall_score(y, y_pred))
        f1score.append(f1_score(y, y_pred))
        
        print("Precision:", precision[-1])
        print("Recall:", recall[-1])
        print("F1 score:", f1score[-1]) 
    
    print(precision)
    print(recall)
    print(f1score)

def evaluate_model(path='./saved_models', namesuffix="", image_range=[150000, 155000]):
    rdf, pixels, offsets = load_model(path, namesuffix)
    y_pred = []
    y = []
    
    step = 200
    for i in range(image_range[0], image_range[1], step):
        depth_images, masks = dataset.load_images(i, i + step)
        X_test, y_test = extract_features_and_labels(depth_images, masks, offsets=offsets, pixels=pixels)
        
        predicted_proba = rdf.predict_proba(X_test)
        predicted_subset = (predicted_proba [:,1] >= 0.5).astype('int')
        y_pred.extend(predicted_subset)
        y.extend(y_test)
        
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1 score:", f1_score(y, y_pred))
          
"""
Detects a hand in the depth image. 
Creates a (2D or 3D?) bounding box containing the detected hand.
Returns a subimage defined by the bounding box.

Returns a subimage of the detected hand from the original depth image.
The returned image properties:
    Default Size = 128 x 128
    Values = [-1, 1]
    Background values = 1
"""

def predict_boundary(model, pixels, offsets, depth_image, target_size = (128, 128)):
    depth_image = depth_image.astype(int)
    depth_image[depth_image == 0] = HUGE_INT
    depth_image = depth_image.astype(np.float32)
    print(depth_image)
    X = extract_features(depth_image[np.newaxis,:], offsets=offsets, pixels=pixels)
    predicted_proba = model.predict_proba(X)
    predicted = (predicted_proba[:,1] >= 0.35).astype('int')
    
    hand = [pix for pred, pix in zip(predicted, pixels) if pred == 1]
    hand = np.array(hand)
    
    # normalize image, convert to rgb
    depth_image -= depth_image.min()
    depth_image[depth_image > 10000.0] = 0
    depth_image *= 1.0/depth_image.max()
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
    depth_image = 1 - depth_image 
    
    # draw predicted hand pixels on the depth map
    plt.imshow(depth_image, vmin=0, vmax=255);
    if (len(hand) > 0):
        plt.scatter(hand[:,1], hand[:,0], c='r', s=3)
    plt.title('Depth Image'); 
    plt.show()
    


if __name__ == "__main__":
    #create_train_incrementally_save(500, 5, 22, "_50k", first_image_index=0, last_image_index=50000)    
    #evaluate_model(namesuffix="20200726_233534", image_range=[150000, 150100])
    #predict_all_pixels()
    create_train_save()

