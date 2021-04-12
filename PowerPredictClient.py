# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import time

import grpc
from concurrent import futures
from proto import powerpredict_pb2, powerpredict_pb2_grpc

if __name__ == "__main__":
    channel = grpc.insecure_channel("localhost:50051")
    # channel = grpc.insecure_channel("localhost:50050")
    # channel = grpc.insecure_channel("172.28.0.77:50050")
    stub = powerpredict_pb2_grpc.PowerPredictServiceStub(channel)

    # res = stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="compute01", start="1568530100", end="1568530109", algorithm="LSTMx2"))
    # res = stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="compute01", start="1568510400", end="1568510409", algorithm="ARIMA"))
    # res = stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="compute01", start="1568530100", end="1568530109", algorithm="tcn"))
    # res = stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="compute01", start="1568510400", end="1568510409", algorithm="darnn"))
    # res = stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="compute01", start="1568294221", end="1568440030",
    #                                                              algorithm="rf", type="server"))
    # res = stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="compute01", start="1568510700", end="1568510709", algorithm="empty"))
    # pod 能耗预测
    res = stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="linpack11", start="1568440021", end="1568440030",
                                                                 algorithm="rf", type="pod"))
    print(res)

    # for i in range(10):
    #    start = 1568297400
    #    res = stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="compute01", start=str(start+i),
    #                                                                 end=str(start+10+i), algorithm="rf", type="server"))
    #    print(res)
