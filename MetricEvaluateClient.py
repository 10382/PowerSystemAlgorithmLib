# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import time

import grpc
from concurrent import futures
from proto import metricevaluate_pb2, metricevaluate_pb2_grpc

if __name__ == "__main__":
    channel = grpc.insecure_channel("localhost:50053")
    stub = metricevaluate_pb2_grpc.MetricEvaluateServiceStub(channel)


    # 硬件能耗分解请求
    # res = stub.MetricEvaluate(metricevaluate_pb2.MetricEvaluateRequest(host="compute01", type="qos", start="1",
    #                                                                    end="2", algorithm="brb"))
    res = stub.MetricEvaluate(metricevaluate_pb2.MetricEvaluateRequest(type="dc", host="datacenter", start="23",
                                                                       end="34", algorithm="membership"))
    print(res)
