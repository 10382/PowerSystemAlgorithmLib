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
from proto import metricevaluate_pb2, metricevaluate_pb2_grpc
from utils import redis_utils, preprocess
from algorithm.metric.qos_brb.evaluate import main_opt_bayes
from algorithm.metric.dc_level.evaluate import sample_dc_levels

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
SEQ_LEN = 10

REDIS_ADDR = os.getenv("REDIS_IP")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PWD = os.getenv("REDIS_PASSWORD")
REDIS_DB = os.getenv("REDIS_DATABASE")
# QoS 指标评估数据读取
QOS_BRB_PARH = "algorithm/metric/qos_brb/compute01.xlsx"
QOS_BRB_DATA = pd.read_excel(QOS_BRB_PARH, header=None)

class MetricEvaluateService(metricevaluate_pb2_grpc.MetricEvaluateServiceServicer):
    # @profile
    def MetricEvaluate(self, request, context):
        # 如果为 qos 指标评估
        if request.type == "qos":
            # 且算法为 brb 算法
            if request.algorithm == "brb":
                ret = main_opt_bayes(QOS_BRB_DATA.iloc[int(request.start):int(request.end), :],
                                     int(request.start), int(request.end))
                return metricevaluate_pb2.MetricEvaluateReply(metrics="%s" % {"qos": ret})
        # 如果为数据中心指标评估
        elif request.type == "dc":
            # 如果为最大隶属度算法
            if request.algorithm == "membership":
                ret = sample_dc_levels(int(request.start), int(request.end))
                print(len(ret))
                return metricevaluate_pb2.MetricEvaluateReply(metrics="%s" % {"dc": ret})
        return metricevaluate_pb2.MetricEvaluateReply(metrics="")

# @profile
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    metricevaluate_pb2_grpc.add_MetricEvaluateServiceServicer_to_server(MetricEvaluateService(), server)
    # QOS_BRB_DATA = pd.read_excel(QOS_BRB_PARH, header=None)
    server.add_insecure_port('[::]:50053')
    # server.wait_for_termination(timeout=20)
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
