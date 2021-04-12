# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import grpc
import scipy
import redis
import pickle
import numpy as np
import pandas as pd
from proto import powerevaluate_pb2, powerevaluate_pb2_grpc
from concurrent import futures
from utils import redis_utils, preprocess
from algorithm.decompose import cpu, regtree, hardware
from algorithm.rf import rf_train

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
REDIS_ADDR = os.getenv("REDIS_IP")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PWD = os.getenv("REDIS_PASSWORD")
REDIS_DB = os.getenv("REDIS_DATABASE")
LABEL = "pdu.power"
VM_CONF = {'1': {"cpu.usage.percent": 1, "memory.used.percent": 1}, '2': {"cpu.usage.percent": 2, "memory.used.percent": 2},
           '3': {"cpu.usage.percent": 4, "memory.used.percent": 4}, '4': {"cpu.usage.percent": 8, "memory.used.percent": 8}}
MINMAX_COLS = ["cpu.usage.percent", "memory.used.percent", "disk.vda.disk_octets.write",
               "disk.vda.disk_octets.read", "interface.eth0.if_octets.tx", "interface.eth0.if_octets.rx"]
# pod 特征名
POD_COLS = ["cpu.percent", "memory.percent", "blkio.io_service_bytes_recursive-253-0.write",
            "blkio.io_service_bytes_recursive-253-0.read", "network.usage.tx_bytes", "network.usage.rx_bytes"]
VM_CPU_METRIC = "cpu.usage.percent"
VM_MEM_METRIC = "memory.used.percent"
TYPE_COL_NAME = "type"
CPU_COUNT = 4
MEM_CAP = 8
MINMAX_RANGE = [[0, 0, 0, 0, 0, 0], [100 * CPU_COUNT, 100 * MEM_CAP, 1e9, 1e9, 1e9, 1e9]]
MODEL_PATH = {"rf": ("pickle", "model/rf.pkl"), "lstmx2": ("torch", "model/lstmx2.pkl"), "tcn": ("torch", "model/tcn.pkl"),
              "darnn": ("torch", "model/darnn.pkl"), "direct": ("pickle", "model/rf-pod.pkl")}
MODEL_DICT = {"cpu.usage.percent": 4, "memory.used.percent": 1}

class PowerEvaluateService(powerevaluate_pb2_grpc.PowerEvaluateServiceServicer):

    def PowerEvaluate(self, request, context):
        # # print("服务端接收到用户请求："+request.host)
        # # 根据请求的 targetTs 去向前取训练数据
        # start_ts = str(int(request.targetTimestamp) - 19999)
        # end_ts = request.targetTimestamp
        # # 根据时间戳判断在哪个时间段内
        # redis_req_list = redis_utils.redis_ts_split(start_ts, end_ts, request.host)
        # print(redis_req_list)
        # # Redis 连接配置
        # redis_pool = redis.ConnectionPool(host=REDIS_ADDR, port=6379, password="lidata429", db=2)
        # # 建立连接
        # r = redis.Redis(connection_pool=redis_pool)
        # # 请求虚拟机 dict
        # vm_list = preprocess.req_vmlist(redis_req_list, r, request.targetTimestamp, start_ts, end_ts)
        # vm_df_dict = dict()
        # # 得到每个虚拟机的 dataframe 数据
        # for vm_name in vm_list:
        #     vm_df_dict[vm_name] = preprocess.req2df(redis_utils.redis_ts_split(start_ts, end_ts, vm_name),
        #                                             r, start_ts, end_ts)
        #     # 先填充一下虚拟机类型（有该列时删除）
        #     vm_df_dict[vm_name][TYPE_COL_NAME] = '2'
        # # print(vm_df_dict)
        # # 显示 dataframe 的所有列
        # pd.set_option('display.max_columns', None)
        # # print(vm_df_dict)
        # # 测试请求内容
        # print("host: %s, hostType: %s, start: %s, end: %s, algorithm: %s" %
        #       (request.host, request.hostType, request.start, request.end, request.algorithm))
        # return powerevaluate_pb2.PowerEvaluateReply(power='123')

        # Redis 连接配置
        redis_pool = redis.ConnectionPool(host=REDIS_ADDR, port=REDIS_PORT, password=REDIS_PWD, db=REDIS_DB)
        # 建立连接
        r = redis.Redis(connection_pool=redis_pool)

        # 对于训练来说的话，传入的 host 就是物理机名
        server_id = request.host

        # 根据时间戳判断在哪个时间段内去计算有关的物理机请求有哪些
        redis_req_list = redis_utils.redis_ts_split(request.start, request.end, server_id)
        print(redis_req_list)

        # 如果是 pod 能耗评估
        if request.hostType == "pod":
            # 读取 pod 相关数据
            pod_df = preprocess.req2df(redis_req_list, r, request.start, request.end)
            pod_df = pod_df[pod_df.power.notna()]
            pod_df = preprocess.fillna_decompose(pod_df)
            pod_cols = list(map(lambda x: "docker." + request.host + "." + x, POD_COLS))
            rf_train(pod_df.loc[:, pod_cols].values, pod_df.loc[:, "power"].values, model_path="model/rf-pod.pkl")
            return powerevaluate_pb2.PowerEvaluateReply(power='Success!')

        # 请求虚拟机目标时间戳下的 dict
        vm_list = preprocess.req_vmlist(redis_req_list, r, request.start, request.end)
        vm_df_dict = dict()
        # 得到每个虚拟机的 dataframe 数据
        for vm_name in vm_list:
            vm_df_dict[vm_name] = preprocess.req2df(redis_utils.redis_ts_split(request.start, request.end, vm_name),
                                                    r, request.start, request.end)
            # 先填充一下虚拟机类型（有该列时删除）
            vm_df_dict[vm_name][TYPE_COL_NAME] = '2'
        # 显示 dataframe 的所有列
        pd.set_option('display.max_columns', None)
        # print(vm_df_dict)
        # 获取当前时刻的能耗值
        server_df = preprocess.req2df(redis_req_list, r, request.start, request.end)
        pdu_df = server_df.loc[:, [LABEL]]
        # 如果是请求的是虚拟机分解且采用的是CPU使用率能耗分解模型
        if request.hostType == "vm" and request.algorithm == "cpu":
            # 根据 CPU 使用率分解到功率
            vm_power_dict = cpu.decompose(vm_df_dict, VM_CPU_METRIC, pdu_df)
            return powerevaluate_pb2.PowerEvaluateReply(power='%s' % (vm_power_dict))

        # 其他情况，需要计算虚拟机类型对应的特征和
        vm_type_sum_df = preprocess.vm_type_sum(vm_df_dict.values(), VM_CONF, multi_gb_names=["index", "type"])
        # vm_sum_df = preprocess.vm_type_sum(vm_df_dict.values(), VM_CONF)
        # 根据虚拟机类型去做特征值的放大
        vm_type_sum_df = preprocess.vm_type_enlarge(vm_type_sum_df, VM_CONF, cpu_metric=None, mem_metric=VM_MEM_METRIC,
                                                    minmax_columns=MINMAX_COLS, minmax_range=MINMAX_RANGE)
        # 单虚拟机值放大+归一化
        for vm_name in vm_list:
            vm_df_dict[vm_name] = preprocess.vm_type_enlarge(vm_df_dict[vm_name].reset_index().set_index(TYPE_COL_NAME),
                                                             VM_CONF, cpu_metric=None, mem_metric=VM_MEM_METRIC,
                                                             minmax_columns=MINMAX_COLS, minmax_range=MINMAX_RANGE)
        # 按虚拟机类型合并为单个 dataframe(将 type 列扩充到特征上)
        vm_train_df, vm_cpu_sum_df = preprocess.vm_type_concat(vm_type_sum_df, "index", VM_CONF.keys(),
                                                               request.start, request.end,
                                                               resort_columns=["index"] + MINMAX_COLS,
                                                               cpu_column=VM_CPU_METRIC)
                                                               # cpu_column=None)
        # cpu_column=VM_CPU_METRIC)
        if request.hostType == "vm":
            if request.algorithm == "regtree":
                # print(cpu)
                # return
                print(vm_train_df.nunique())
                # print(vm_train_df)
                # # 缺失值填充
                # preprocess.fillna_decompose(vm_train_df, methods="interpolate")
                # # 异常值处理
                # preprocess.outlier_decompose(vm_train_df)
                merge_df = pd.concat((vm_train_df, pdu_df), axis=1).dropna()
                # print(merge_df)
                left_cols = list(merge_df.columns)
                left_cols.remove(LABEL)
                print(left_cols)
                X = merge_df.loc[:, left_cols].values
                y = merge_df.loc[:, LABEL].values
                # print(scipy.stats.describe(X, axis=1))
                print("X shape:", X.shape)
                print("y shape:", y.shape)
                rt = regtree.RegTree()
                rt.train(X, y)
                pick = pickle.dumps(rt)
                file = open("model/decompose/regtree-printWb.pkl", 'wb')
                file.write(pick)
                file.close()
        elif request.hostType == "hardware":
            if request.algorithm == "xgboost":
                print("Hardware: xgboost")
                hardware.xgboost_train(vm_train_df, pdu_df, model_path="model/decompose/xgboost.pkl")
#         归一化
#         minmax_np = preprocess.minmax(vm_sum_df, columns=MINMAX_COLS, range=MINMAX_RANGE)
#         flatten_np = minmax_np.flatten()
#        # pred.rf_train(minmax_np, pdu_df.values[:minmax_np.shape[0]])
#        y_pred = pred.rf_test(minmax_np)
#        # return powerpredict_pb2.PowerPredictReply(power='The power of %s is %s !' % (request.host, y_pred[0]))
        return powerevaluate_pb2.PowerEvaluateReply(power='%s' % ("123"))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    powerevaluate_pb2_grpc.add_PowerEvaluateServiceServicer_to_server(PowerEvaluateService(), server)
    server.add_insecure_port('[::]:50012')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
