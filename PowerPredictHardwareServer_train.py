# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import time

import grpc
import redis
import numpy as np
import pandas as pd
from concurrent import futures
from proto import powerpredict_pb2, powerpredict_pb2_grpc
from utils import redis_utils, preprocess
from algorithm import pred, lstmx2, tcn, darnn

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
SEQ_LEN = 10
REDIS_ADDR = "10.160.109.63"
LABEL = "pdu.power"
TOTAL_COLS = ["cpu.active.percent", "memory.used.percent", "disk.sda.disk_octets.write",
               "disk.sda.disk_octets.read", "interface.eth0.if_octets.tx", "interface.eth0.if_octets.rx",
               "pdu.power"]
MERGE_COLS = ["cpu.active.percent", "memory.used.percent", "disk.sda.disk_octets",
              "interface.eth0.if_octets"]
# MINMAX_RANGE = [[0, 100], [0, 100], [0, 1e9], [0, 1e9], [0, 1e9], [0, 1e9], [10, 75]]
MINMAX_RANGE = [[0, 0, 0, 0], [100, 100, 1e9, 1e9]]

class PowerPredictService(powerpredict_pb2_grpc.PowerPredictServiceServicer):
    @profile
    def PowerPredict(self, request, context):
#        try:
        # print("服务端接收到用户请求："+request.host)
        # 根据请求的start和end以及数据库的数据存储间隔进行请求的划分
        # 服务器相关的请求
        redis_req_list = redis_utils.redis_ts_split(request.start, request.end, request.host)
        print(redis_req_list)
        # # PDU 相关的请求
        # pdu_req_list = redis_utils.redis_ts_split(request.start, request.end, "pdu-mini")
        # Redis 连接配置
        redis_pool = redis.ConnectionPool(host=REDIS_ADDR, port=6379, password="lidata429", db=1)
        # 建立连接
        r = redis.Redis(connection_pool=redis_pool)
        # 请求数据并转datafrme
        server_df = preprocess.req2df(redis_req_list, r, request.start, request.end)
        pdu_df = server_df.loc[:, [LABEL]]

        # # 显示所有列
        # pd.set_option('display.max_columns', None)
        # # 查看全量数据上的特征范围，来作为归一化范围
        # print(server_df.describe())
        # return powerpredict_pb2.PowerPredictReply(power='Trained')
        # 缺失值填充
        preprocess.fillna_decompose(server_df)
        # 异常值处理
        preprocess.outlier_decompose(server_df)
        # 归一化
        minmax_np = preprocess.minmax(server_df, MINMAX_COLS, MINMAX_RANGE)
        print(minmax_np)
        print("Metric shape after minmax:", minmax_np.shape)
        minmax_np, y_np = preprocess.pred_concat(minmax_np, pdu_df.values, time_len=SEQ_LEN, train=True)
        print("Metric shape after concat:", minmax_np.shape)
        print("Y shape:", y_np.shape)
#         return powerpredict_pb2.PowerPredictReply(power='Trained')
        print(request.algorithm)
        # print(minmax_np)
        # print(y_np)
        if request.algorithm == "RF" or request.algorithm == "rf":
            pred.rf_train(minmax_np, np.squeeze(y_np))
        if request.algorithm == "ARIMA" or request.algorithm == "arima":
            pred.arima_forecast(pdu_df.values)
        elif request.algorithm == "LSTMx2":
            lstmx2.train(minmax_np, np.squeeze(y_np))
        elif request.algorithm == "TCN" or request.algorithm == "tcn":
            tcn.train(minmax_np, np.squeeze(y_np))
        elif request.algorithm == "DARNN" or request.algorithm == "darnn":
            darnn.train(minmax_np, np.squeeze(y_np))
        # pred.rf_train(minmax_np, pdu_df.values[:minmax_np.shape[0]])
        # y_pred = pred.rf_test(minmax_np)
        # return powerpredict_pb2.PowerPredictReply(power='The power of %s is %s !' % (request.host, y_pred[0]))
        return powerpredict_pb2.PowerPredictReply(power='Trained')
#        except Exception as e:
#            print(e)
#            return powerpredict_pb2.PowerPredictReply(power="")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    powerpredict_pb2_grpc.add_PowerPredictServiceServicer_to_server(PowerPredictService(), server)
    server.add_insecure_port('[::]:50053')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
