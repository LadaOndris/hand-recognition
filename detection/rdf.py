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

def load_dataset(images, test_size = 0.2):
    # X contains features, y contains labels
    features, labels = dataset.get_samples(images, sampled_pixels_count=400, total_features=400)
    # split data in train and test set
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test_size)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # train random forest
    rdf = RandomForestClassifier(max_depth=20, bootstrap=True, n_estimators=64)
    rdf.fit(X_train, y_train)
    return rdf

def save_model(model):
    timestamp = get_current_timestamp()
    dump(model, F"rdf_{timestamp}.joblib") 
    dump(dataset.offsets, F"offsets_{timestamp}.joblib")
    dump(dataset.pixels, F"pixels_{timestamp}.joblib")
    
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
    X_train, X_test, y_train, y_test = load_dataset(images = 200, test_size=0.1)
    print("Training model", flush=True)
    model = train_model(X_train, y_train)
    print("Saving model", flush=True)
    save_model(model)
    print("Evaluating model", flush=True)
    evaluate(model, X_test, y_test)

def load_evaluate():
    print("Loading model", flush=True)
    model = load_model('/home/lada/projects/IBT/detection', '20200712_175549')
    print("Loading dataset", flush=True)
    X_test, y_test = dataset.get_samples(20)
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
    
create_train_save()

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













