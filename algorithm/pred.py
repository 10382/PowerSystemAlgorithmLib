import sys
import json
import redis
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import arma_order_select_ic

metrics = ["cpu.usage", "memory.usage"]

# 选择训练模型
def model_sel(X, y, algorithm="rf", model=None):
    if algorithm == "arima" or algorithm == "ARIMA":
        pass
    return

# def train(X, y, model_path="./rf.pkl"):
def rf_train(X, y, model_path="model/rf.pkl"):
    rfr = RandomForestRegressor(max_depth=10, min_samples_leaf=5)
    rfr.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(rfr, f)

# @profile
# def test(X, model_path="./rf.pkl"):
def rf_test(X, model=None, model_path="model/rf.pkl"):
    if model == None:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    return model.predict(X)[0]

# ARIMA 预测
# @profile
def arima_forecast(y_hist):
    warnings.filterwarnings('ignore')
    # 将数据拍扁
    y_hist = np.squeeze(y_hist).astype("float64")
    # # 自动确定ARMA的pq参数
    # pq_bic = arma_order_select_ic(y_hist, max_ar=5, max_ma=5, ic=['aic', 'bic'])
    # print(pq_bic["bic_min_order"])
    # # 实例化ARMA模型
    # model = ARMA(y_hist, pq_bic.bic_min_order).fit(disp=0, method="css")
    model = ARMA(y_hist, (0, 2)).fit(disp=0, method="css")
    # 预测下一个时间粒度
    out = model.forecast()
    # 取出预测值
    y_pred = out[0]
    return y_pred[0]
