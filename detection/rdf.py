"""
Created on Tue Jul  7 21:11:25 2020

@author: Ladislav Ondris
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from datasets.handseg150k.dataset import load_images
from joblib import dump, load
from datetime import datetime
import numpy as np

def get_current_timestamp():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    return dt_string

def get_serialized_model_filename():
    return F"rdf_{get_current_timestamp()}.joblib"

def feature(depth, x, u, v):
    return
    
def get_features_and_label_for_pixel(image, mask, pixelPos):
    return

def generate_features_and_labels_for_image(image, mask):
    X = np.array()
    y = np.array()
    for row in len(image):
        for col in len(image[0]):
            features, label = get_features_and_label_for_pixel(image, mask, (row, col))
            X = np.concatenate((X, features))
            y = np.concatenate((y, label))
    return X, y

def generate_X_and_y(images, masks):
    X = np.array()
    y = np.array()
    for image, mask in zip(images, masks):
        X_im, y_im = generate_features_and_labels_for_image(image, mask)
        X = np.concatenate((X, X_im))
        y = np.concatenate((y, y_im))
    return X, y

def load_dataset():
    # load multiple images from dataset
    images, masks = load_images(1)
    # for each pixel in each image generate features and corresponding label
    # X contains features, y contains labels
    X, y = generate_X_and_y(images, masks)
    # split data in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # train random forest
    rdf = RandomForestClassifier(max_depth=10, bootstrap=True)
    rdf.fit(X_train, y_train)
    return rdf

def save_model(model):
    dump(model, get_serialized_model_filename()) 

def evaluate(model, X_test, y_test):
    # determine the accuracy predicted by the model
    score = model.score(X_test, y_test)
    print(score)

"""
X_train, X_test, y_train, y_test = load_dataset()
model = train_model(X_train, y_train)
save_model(model)
evaluate(model, X_test, y_test)
"""