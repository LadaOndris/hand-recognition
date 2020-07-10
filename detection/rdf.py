"""
Created on Tue Jul  7 21:11:25 2020

@author: Ladislav Ondris
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from datasets.handseg150k.dataset import get_samples
from joblib import dump, load
from datetime import datetime
import numpy as np

def get_current_timestamp():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    return dt_string

def get_serialized_model_filename():
    return F"rdf_{get_current_timestamp()}.joblib"

def load_dataset(test_size = 0.2):
    # X contains features, y contains labels
    features, labels = get_samples(1)
    # split data in train and test set
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test_size)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # train random forest
    rdf = RandomForestClassifier(max_depth=10, bootstrap=True,)
    rdf.fit(X_train, y_train)
    return rdf

def save_model(model):
    dump(model, get_serialized_model_filename()) 
    
def load_model():
    return load('/home/lada/projects/IBT/detection/rdf_20200710_225703.joblib')

def evaluate(model, X_test, y_test):
    # determine the accuracy predicted by the model
    threshold = 0.5
    predicted_proba = model.predict_proba(X_test)
    predicted = (predicted_proba [:,1] >= threshold).astype('int')
    
    print("Total positive:", sum([l for l in y_test if l == 1]))
    print("Precision:", precision_score(y_test, predicted))
    print("Recall:", recall_score(y_test, predicted))
    print("F1 score:", f1_score(y_test, predicted))
    
def create_train_save():
    print("Loading dataset")
    X_train, X_test, y_train, y_test = load_dataset()
    print("Training model")
    model = train_model(X_train, y_train)
    print("Saving model")
    save_model(model)
    print("Evaluating model")
    evaluate(model, X_test, y_test)

def load_evaluate():
    print("Loading dataset")
    _, X_test, _, y_test = load_dataset(test_size = 0.9)
    print("Loading model")
    model = load_model()
    print("Evaluating model")
    evaluate(model, X_test, y_test)

load_evaluate()