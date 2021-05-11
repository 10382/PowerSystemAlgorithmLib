# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import time

import grpc
from concurrent import futures
from proto import AlgorithmSupport_pb2, AlgorithmSupport_pb2_grpc

if __name__ == "__main__":
    channel = grpc.insecure_channel("localhost:50050")
    stub = AlgorithmSupport_pb2_grpc.AlgorithmSupportServiceStub(channel)

    # # 能耗评估请求
    # res = stub.AlgorithmSupport(AlgorithmSupport_pb2.AlgorithmSupportRequest(entityID="compute01", serviceType="vm", startTimestamp="1568440031",
    #                                                                 endTimestamp="1568440031", algorithm="regtree"))
    # pod 分解相关请求
    res = stub.AlgorithmSupport(
        AlgorithmSupport_pb2.AlgorithmSupportRequest(entityID="linpack11", serviceType="pod_e", startTimestamp="1568440021",
                                               endTimestamp="1568440030", algorithm="regtree"))
    # res = stub.AlgorithmSupport(AlgorithmSupport_pb2.AlgorithmSupportRequest(entityID="linpack11", serviceType="pod_e", startTimestamp="1568440021",
    #                                                                 endTimestamp="1568440030", algorithm="direct"))
    # # 硬件能耗分解请求
    # res = stub.AlgorithmSupport(AlgorithmSupport_pb2.AlgorithmSupportRequest(entityID="compute01", serviceType="hardware", startTimestamp="1568510603",
    #                                                                 endTimestamp="1568510606", algorithm="xgboost"))
    # # 硬件能耗分解请求
    # res = stub.AlgorithmSupport(AlgorithmSupport_pb2.AlgorithmSupportRequest(serviceType="qos", entityID="compute01",
    #                                                                          startTimestamp="4", endTimestamp="6",
    #                                                                          algorithm="brb"))
    # res = stub.AlgorithmSupport(AlgorithmSupport_pb2.AlgorithmSupportRequest(serviceType="dc", entityID="datacenter",
    #                                                                          startTimestamp="23", endTimestamp="34",
    #                                                                          algorithm="membership"))
    # 能耗预测请求
    # res = stub.AlgorithmSupport(AlgorithmSupport_pb2.AlgorithmSupportRequest(entityID="compute01", startTimestamp="1568294221",
    #                                                                          endTimestamp="1568440030", algorithm="rf",
    #                                                                          serviceType="server"))
    # res = stub.AlgorithmSupport(AlgorithmSupport_pb2.AlgorithmSupportRequest(entityID="compute01", startTimestamp="1568294221",
    #                                                                          endTimestamp="1568440030", algorithm="lstmx2",
    #                                                                          serviceType="server"))
    # res = stub.AlgorithmSupport(
    #     AlgorithmSupport_pb2.AlgorithmSupportRequest(entityID="compute01", startTimestamp="1568294221",
    #                                                  endTimestamp="1568440030", algorithm="tcn",
    #                                                  serviceType="server"))
    # res = stub.AlgorithmSupport(
    #     AlgorithmSupport_pb2.AlgorithmSupportRequest(entityID="compute01", startTimestamp="1568294221",
    #                                                  endTimestamp="1568440030", algorithm="arima",
    #                                                  serviceType="server"))
    # res = stub.AlgorithmSupport(
    #     AlgorithmSupport_pb2.AlgorithmSupportRequest(entityID="compute01", startTimestamp="1568294221",
    #                                                  endTimestamp="1568440030", algorithm="darnn",
    #                                                  serviceType="server"))
    # res = stub.AlgorithmSupport(AlgorithmSupport_pb2.AlgorithmSupportRequest(entityID="compute01", startTimestamp="1568510700", endTimestamp="1568510709", algorithm="empty"))
    # # pod 能耗预测
    # res = stub.AlgorithmSupport(AlgorithmSupport_pb2.AlgorithmSupportRequest(entityID="linpack11", startTimestamp="1568440021", endTimestamp="1568440030",
    #                                                              algorithm="rf", serviceType="pod"))
    print(res)
