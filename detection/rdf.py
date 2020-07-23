"""
Created on Tue Jul  7 21:11:25 2020

@author: Ladislav Ondris
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from datasets.handseg150k.dataset import HandsegDataset, HUGE_INT
from joblib import dump, load
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2

dataset = HandsegDataset()

def get_current_timestamp():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    return dt_string

def load_dataset(start_index, end_index, test_size = 0.2):
    # X contains features, y contains labels
    features, labels = dataset.get_samples(start_index, end_index)
    # split data in train and test set
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test_size)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # train random forest
    rdf = RandomForestClassifier(max_depth=20, bootstrap=True, n_estimators=64, n_jobs=-1)
    rdf.fit(X_train, y_train)
    return rdf

def save_model(model, path):
    timestamp = get_current_timestamp()
    dump(model, os.path.join(path, F"rdf_{timestamp}.joblib")) 
    dump(dataset.offsets, os.path.join(path, F"offsets_{timestamp}.joblib"))
    dump(dataset.pixels, os.path.join(path, F"pixels_{timestamp}.joblib"))
    
def load_model(path, timestamp):
    dataset.offsets = load(os.path.join(path, F"offsets_{timestamp}.joblib"))
    dataset.pixels = load(os.path.join(path, F"pixels_{timestamp}.joblib"))
    return load(os.path.join(path, F"rdf_{timestamp}.joblib"))

def evaluate(model, X_test, y_test):
    # determine the accuracy predicted by the model
    threshold = 0.5
    predicted_proba = model.predict_proba(X_test)
    predicted = (predicted_proba [:,1] >= threshold).astype('int')
    
    positive_pixels = sum([l for l in y_test if l == 1])
    print("Total positive:", positive_pixels, "out of", len(y_test), "pixels")
    print("Precision:", precision_score(y_test, predicted))
    print("Recall:", recall_score(y_test, predicted))
    print("F1 score:", f1_score(y_test, predicted)) 
    
def create_train_save():
    print("Loading dataset", flush=True)
    X_train, X_test, y_train, y_test = load_dataset(0, 199, test_size=0.1)
    print("Training model", flush=True)
    model = train_model(X_train, y_train)
    print("Saving model", flush=True)
    save_model(model, './saved_models')
    print("Evaluating model", flush=True)
    evaluate(model, X_test, y_test)
    
def create_train_incrementally_save():
    rdf = RandomForestClassifier(max_depth=20, bootstrap=True, n_estimators=1, 
                                 n_jobs=-1, warm_start = True)
    step = 200
    for i in range(0, 5000, step):
        print(F"Loading dataset {i}", flush=True)
        X, y = dataset.get_samples(i, i + step - 1, total_features = 200)
        print("Training model", flush=True)
        rdf.n_estimators += 1
        rdf.fit(X, y)
    print("Saving model", flush=True)
    save_model(rdf, './saved_models')

"""
Good models: 20200712_175549, 20200712_231245
"""
def load_evaluate():
    print("Loading model", flush=True)
    model = load_model('/home/lada/projects/IBT/detection', '20200713_105103')
    print("Loading dataset", flush=True)
    X_test, y_test = dataset.get_samples(5000, 5199)
    print("Evaluating model", flush=True)
    evaluate(model, X_test, y_test)


def predict_all_pixels():
    model = load_model('/home/lada/projects/IBT/detection', '20200712_231245')
    depth_image, mask = dataset.load_image(4)
    depth_image = depth_image.astype(np.float32)
    
    print(np.max(depth_image))
    X, y = dataset.get_samples_for_image(depth_image, mask)
    print(X.shape)
    predicted_proba = model.predict_proba(X)
    predicted = (predicted_proba[:,1] >= 0.5).astype('int')
    print("Precision:", precision_score(y, predicted))
    hand = [pix for pred, pix in zip(predicted, dataset.pixels) if pred == 1]
    hand = np.array(hand)
               
    # normalize image, convert to rgb
    depth_image -= depth_image.min()
    depth_image[depth_image > 10000.0] = 0
    depth_image *= 1.0/depth_image.max()
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
    depth_image = 1 - depth_image 
    
    # draw predicted hand pixels on the depth map
    plt.imshow(depth_image, vmin=0, vmax=255);
    plt.scatter(hand[:,1], hand[:,0], c='r', s=3)
    plt.title('Depth Image'); plt.show()
    #plt.imshow(mask, cmap='BuPu'); plt.title('Mask Image'); plt.show()
    

def evaluate_image():
    model = load_model('/home/lada/projects/IBT/detection', '20200712_175549')
    image, mask = dataset.load_image(28000)
    X, y = dataset.get_samples_for_image(image, mask)
    evaluate(model, X, y)
    
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
def predict_boundary(depth_image, target_size = (128, 128)):
    
    return
    
#load_evaluate()

"""
import pyrealsense2 as rs

pipe = rs.pipeline()
profile = pipe.start()
try:
  for i in range(0, 100):
    frames = pipe.wait_for_frames()
    for f in frames:
      print(f.profile)
finally:
    pipe.stop()
"""













