# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import time

import grpc
from concurrent import futures
from proto import powerevaluate_pb2, powerevaluate_pb2_grpc

if __name__ == "__main__":
    channel = grpc.insecure_channel("localhost:50052")
    stub = powerevaluate_pb2_grpc.PowerEvaluateServiceStub(channel)

    # res = stub.PowerEvaluate(powerevaluate_pb2.PowerEvaluateRequest(host="compute01", hostType="vm", start="1568440031",
    #                                                                 end="1568440031", algorithm="regtree"))
    # res = stub.PowerEvaluate(powerevaluate_pb2.PowerEvaluateRequest(host="compute01", targetTimestamp="1568295501", algorithm="cpu"))
    # pod 分解相关请求
    res = stub.PowerEvaluate(powerevaluate_pb2.PowerEvaluateRequest(host="linpack11", hostType="pod", start="1568440021",
                                                                    end="1568440030", algorithm="regtree"))
    # res = stub.PowerEvaluate(powerevaluate_pb2.PowerEvaluateRequest(host="linpack11", hostType="pod", start="1568440021",
    #                                                                 end="1568440030", algorithm="direct"))
    # 硬件能耗分解请求
    # res = stub.PowerEvaluate(powerevaluate_pb2.PowerEvaluateRequest(host="compute01", hostType="hardware", start="1568510603",
    #                                                                 end="1568510606", algorithm="xgboost"))
    print(res)

    # # 循环测试
    # start = time.time()
    # for i in range(10000):
    #     ts = str(1568294223 + i)
    #     res = stub.PowerEvaluate(powerevaluate_pb2.PowerEvaluateRequest(host="compute01", targetTimestamp=ts,
    #                                                                     algorithm="regtree"))
    #     print(res)
    # total_time = time.time() - start
    # print("运行总时间为:", total_time, "s,\n平均单次请求时间为:", total_time / 10000)
