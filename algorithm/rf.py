import sys
import json
import redis
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# def train(X, y, model_path="./rf.pkl"):
def rf_train(X, y, model_path="model/rf.pkl"):
    rfr = RandomForestRegressor()
    rfr.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(rfr, f)

# def test(X, model_path="./rf.pkl"):
def rf_test(X, model_path="model/rf.pkl", model=None):
    if model is None:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    return model.predict(X)


