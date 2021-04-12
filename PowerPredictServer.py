# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import grpc
import redis
import torch
import pickle
import numpy as np
import pandas as pd
from concurrent import futures
from proto import powerpredict_pb2, powerpredict_pb2_grpc
from utils import redis_utils, preprocess
from algorithm import pred, lstmx2, tcn, darnn

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
SEQ_LEN = 10

REDIS_ADDR = os.getenv("REDIS_IP")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PWD = os.getenv("REDIS_PASSWORD")
REDIS_DB = os.getenv("REDIS_DATABASE")
SERVER_COLS = ["cpu.active.percent", "memory.used.percent", "disk.sda.disk_octets.write",
               "disk.sda.disk_octets.read", "interface.eth0.if_octets.tx", "interface.eth0.if_octets.rx",
               "pdu.power"]
SERVER_LABEL = "pdu.power"
POD_COLS = ["cpu.percent", "memory.percent", "blkio.io_service_bytes_recursive-253-0.write",
            "blkio.io_service_bytes_recursive-253-0.read", "network.usage.tx_bytes", "network.usage.rx_bytes",
            "power"]
POD_LABEL = "power"
VM_CONF = {'1': {"cpu.usage.percent": 1, "memory.used.percent": 1}, '2': {"cpu.usage.percent": 2, "memory.used.percent": 2},
           '3': {"cpu.usage.percent": 4, "memory.used.percent": 4}, '4': {"cpu.usage.percent": 8, "memory.used.percent": 8}}
CPU_COUNT = 4
MEM_CAP = 8
MINMAX_RANGE = [[0, 0, 0, 0, 0, 0, 0], [100, 100, 1e9, 1e9, 1e9, 1e9, 75]]  # 功率得从0开始归一化，否则在做容器能耗预测的时候可能会有空值
MODEL_PATH = {"rf": ("pickle", "model/rf.pkl"), "lstmx2": ("torch", "model/lstmx2.pkl"), "tcn": ("torch", "model/tcn.pkl"),
              "darnn": ("torch", "model/darnn.pkl")}
MODEL_DICT = {}

class PowerPredictService(powerpredict_pb2_grpc.PowerPredictServiceServicer):
    # @profile
    def PowerPredict(self, request, context):
        try:
            # 根据预测的是物理机还是虚拟机设置变量
            if request.type == "server":
                MINMAX_COLS = SERVER_COLS
                LABEL = SERVER_LABEL
            else:
                MINMAX_COLS = POD_COLS
                LABEL = POD_LABEL
                TYPE = '2'  # 写死虚拟机类型为2
            # 根据请求的start和end以及数据库的数据存储间隔进行请求的划分
            # # 服务器相关的请求
            # redis_req_list, pdu_req_list = redis_utils.redis_ts_split(request.start, request.end, request.host)
            # print(redis_req_list, pdu_req_list)
            # # PDU 相关的请求
            # pdu_req_list = redis_utils.redis_ts_split(request.start, request.end, "pdu-mini")
            # 【新】服务器相关的请求
            redis_req_list = redis_utils.redis_ts_split(request.start, request.end, request.host)
            print(redis_req_list)
            # Redis 连接配置
            redis_pool = redis.ConnectionPool(host=REDIS_ADDR, port=REDIS_PORT, password=REDIS_PWD, db=REDIS_DB)
            # 建立连接
            r = redis.Redis(connection_pool=redis_pool)
            # # 请求数据并转datafrme
            # server_df = preprocess.req2df(redis_req_list, r, request.start, request.end)
            # pdu_df = preprocess.req2df(pdu_req_list, r, request.start, request.end)
            # server_df = server_df.merge(pdu_df, left_index=True, right_index=True)
            # 【新】请求数据并转datafrme
            server_df = preprocess.req2df(redis_req_list, r, request.start, request.end)
            pdu_df = server_df.loc[:, [LABEL]]
            print("Metric df shape:", server_df.shape, "Power df shape:", pdu_df.shape)
        except Exception as e:
            print("Redis Related Error: %s" % (e))
            return powerpredict_pb2.PowerPredictReply(power='')
        try:
            # 缺失值填充
            preprocess.fillna_decompose(server_df, methods="interpolate")
            # 防止 pod 数据开头与末尾缺失，填充
            if request.type == "pod":
                preprocess.fillna_decompose(server_df, methods="fb")
            # 异常值处理
            preprocess.outlier_decompose(server_df)
            # 归一化
            if request.type == "server":
                minmax_np = preprocess.minmax(server_df, MINMAX_COLS, MINMAX_RANGE)
            else:
                minmax_range = MINMAX_RANGE * np.array([CPU_COUNT / VM_CONF[TYPE]["cpu.usage.percent"],
                                                        MEM_CAP / VM_CONF[TYPE]["memory.used.percent"], 1, 1, 1, 1, 1])
                # 预处理，容器特征名去容器名化
                server_df = preprocess.rmname_from_cols(server_df, "docker." + request.host + '.')
                minmax_np = preprocess.minmax(server_df, MINMAX_COLS, minmax_range)
            print("Metric shape after minmax:", minmax_np.shape)
            minmax_np, y_np = preprocess.pred_concat(minmax_np, pdu_df.values, time_len=SEQ_LEN)
            print("Metric shape after concat:", minmax_np.shape)
    #        print(y_np.shape)
        except Exception as e:
            print("Preprocessing Related Error: %s" % (e))
            return powerpredict_pb2.PowerPredictReply(power='')
        try:
            print(request.algorithm)
            time_start = time.time()
            if request.algorithm == "RF" or request.algorithm == "rf":
                y_pred = pred.rf_test(minmax_np, model=MODEL_DICT["rf"])
            if request.algorithm == "ARIMA" or request.algorithm == "arima":
                y_pred = pred.arima_forecast(pdu_df.values)
            elif request.algorithm == "LSTMx2" or request.algorithm == "lstmx2":
                y_pred = lstmx2.predict(minmax_np, model=MODEL_DICT["lstmx2"])
            elif request.algorithm == "TCN" or request.algorithm == "tcn":
                y_pred = tcn.predict(minmax_np, model=MODEL_DICT["tcn"])
            elif request.algorithm == "DARNN" or request.algorithm == "darnn":
                y_pred = darnn.predict(minmax_np, model=MODEL_DICT["darnn"])
            print("算法运行时长:", time.time() - time_start)
            # pred.rf_train(minmax_np, pdu_df.values[:minmax_np.shape[0]])
            # y_pred = pred.rf_test(minmax_np)
            # return powerpredict_pb2.PowerPredictReply(power='The power of %s is %s !' % (request.host, y_pred[0]))
        except Exception as e:
            print("Algorithm Related Error: %s" % (e))
            return powerpredict_pb2.PowerPredictReply(power='')
        return powerpredict_pb2.PowerPredictReply(power='%s,%s' % (pdu_df.values[-1][0], y_pred))
#        except Exception as e:
#            print(e)
#            return powerpredict_pb2.PowerPredictReply(power="")

# @profile
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    powerpredict_pb2_grpc.add_PowerPredictServiceServicer_to_server(PowerPredictService(), server)
    for model in MODEL_PATH.keys():
        path = MODEL_PATH[model][1]
        print(model, path)
        if MODEL_PATH[model][0] == "pickle":
            with open(path, "rb") as fa:
                MODEL_DICT[model] = pickle.load(fa)
        else:
            MODEL_DICT[model] = torch.load(path)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
