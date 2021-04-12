# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import time

import grpc
from concurrent import futures
from proto import powerpredict_pb2, powerpredict_pb2_grpc

if __name__ == "__main__":
    channel = grpc.insecure_channel("localhost:50061")
    stub = powerpredict_pb2_grpc.PowerPredictServiceStub(channel)

    # stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="compute01", start="1568294221", end="1568560471",
    #                                                        algorithm="LSTMx2"))
    # stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="compute01", start="1568510400", end="1568510799",
    #                                                        algorithm="DARNN"))
    # res = stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="compute01", start="1568294218", end="1568294500", algorithm="ARIMA"))
    # res = stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="compute01", start="1568294221", end="1568374236", algorithm="tcn"))
    # 性能测试用
    res = stub.PowerPredict(
        powerpredict_pb2.PowerPredictRequest(host="compute01", start="1568294221", end="1568560471", algorithm="rf",
                                             type="server"))

#    for i in range(10):
#        start = 1568297400
#        res = stub.PowerPredict(powerpredict_pb2.PowerPredictRequest(host="compute01", start=str(start+i), end=str(start+10+i), algorithm="arima"))
    print(res)

