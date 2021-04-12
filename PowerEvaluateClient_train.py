# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import time

import grpc
from concurrent import futures
from proto import powerevaluate_pb2, powerevaluate_pb2_grpc

if __name__ == "__main__":
    channel = grpc.insecure_channel("localhost:50012")
    stub = powerevaluate_pb2_grpc.PowerEvaluateServiceStub(channel)

    # res = stub.PowerEvaluate(powerevaluate_pb2.PowerEvaluateRequest(host="compute01", hostType="hardware", start="1568294221",
    #                                                                 end="1568314220", algorithm="xgboost"))

    # res = stub.PowerEvaluate(powerevaluate_pb2.PowerEvaluateRequest(host="compute01", hostType="hardware", start="1568294221",
    #                                                                 end="1568314220", algorithm="xgboost"))
    res = stub.PowerEvaluate(powerevaluate_pb2.PowerEvaluateRequest(host="linpack11", hostType="pod", start="1568294221",
                                                                    end="1568560471", algorithm="direct"))
    print(res)

