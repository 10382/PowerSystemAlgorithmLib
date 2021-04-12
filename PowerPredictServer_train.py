# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import grpc
import redis
import numpy as np
import pandas as pd
from concurrent import futures
from proto import powerpredict_pb2, powerpredict_pb2_grpc
from utils import redis_utils, preprocess
from algorithm import pred, lstmx2, tcn, darnn

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
REDIS_ADDR = os.getenv("REDIS_IP")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PWD = os.getenv("REDIS_PASSWORD")
REDIS_DB = os.getenv("REDIS_DATABASE")
SEQ_LEN = 10
LABEL = "pdu.power"
MINMAX_COLS = ["cpu.active.percent", "memory.used.percent", "disk.sda.disk_octets.write",
               "disk.sda.disk_octets.read", "interface.eth0.if_octets.tx", "interface.eth0.if_octets.rx",
               "pdu.power"]
MINMAX_RANGE = [[0, 0, 0, 0, 0, 0, 10], [100, 100, 1e9, 1e9, 1e9, 1e9, 75]]

class PowerPredictService(powerpredict_pb2_grpc.PowerPredictServiceServicer):
    # @profile
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
        redis_pool = redis.ConnectionPool(host=REDIS_ADDR, port=REDIS_PORT, password=REDIS_PWD, db=REDIS_DB)
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
        # 异常值处理
        preprocess.outlier_decompose(server_df)
        # 归一化
        minmax_np = preprocess.minmax(server_df, MINMAX_COLS, MINMAX_RANGE)
        print(minmax_np)
        print("归一化后数据维度:", minmax_np.shape)
        minmax_np, y_np = preprocess.pred_concat(minmax_np, pdu_df.values, time_len=SEQ_LEN, train=True)
        print("时序拼接后数据维度:", minmax_np.shape)
        print("Y的数据维度:", y_np.shape)
        drop_np = np.concatenate((minmax_np, y_np), axis=1)
        print("X，Y拼接的维度:", drop_np.shape)
        # 缺失值丢弃
        drop_np = drop_np[~np.isnan(drop_np).any(axis=1), :]
        # # 将 redis 数据获取后保存到本地
        # np.savetxt("pred_train.csv", drop_np, delimiter=",")
        # return powerpredict_pb2.PowerPredictReply(power='Saved')
        print("丢弃缺失值后的维度:", drop_np.shape)
        minmax_np = drop_np[:, :-1]
        y_np = drop_np[:, -1]
        print("最终X的维度:", minmax_np.shape, "Y的维度:", y_np.shape)
        # return powerpredict_pb2.PowerPredictReply(power='Debugging！')

        print(request.algorithm)
        # print(minmax_np)
        # print(y_np)
        if request.algorithm == "RF" or request.algorithm == "rf":
            pred.rf_train(minmax_np, np.squeeze(y_np))
        elif request.algorithm == "ARIMA" or request.algorithm == "arima":
            pred.arima_forecast(pdu_df.values)
        elif request.algorithm == "LSTMx2" or request.algorithm == "lstmx2":
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
    server.add_insecure_port('[::]:50061')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
