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
from proto import powerevaluate_pb2, powerevaluate_pb2_grpc
from concurrent import futures
from utils import redis_utils, preprocess
from algorithm.decompose import cpu, regtree, hardware
from algorithm.rf import rf_test

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
# 从环境变量中读取单节点 redis 参数
REDIS_ADDR = os.getenv("REDIS_IP")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PWD = os.getenv("REDIS_PASSWORD")
REDIS_DB = os.getenv("REDIS_DATABASE")
LABEL = "pdu.power"
# 不同类型虚拟机配置，用于归一化
VM_CONF = {'1': {"cpu.usage.percent": 1, "memory.used.percent": 1}, '2': {"cpu.usage.percent": 2, "memory.used.percent": 2},
           '3': {"cpu.usage.percent": 4, "memory.used.percent": 4}, '4': {"cpu.usage.percent": 8, "memory.used.percent": 8}}
# 虚拟机特征名
MINMAX_COLS = ["cpu.usage.percent", "memory.used.percent", "disk.vda.disk_octets.write",
               "disk.vda.disk_octets.read", "interface.eth0.if_octets.tx", "interface.eth0.if_octets.rx"]
# pod 特征名
POD_COLS = ["cpu.percent", "memory.percent", "blkio.io_service_bytes_recursive-253-0.write",
            "blkio.io_service_bytes_recursive-253-0.read", "network.usage.tx_bytes", "network.usage.rx_bytes"]
# 虚拟机和 pod 对应指标名
VM_CPU_METRIC = "cpu.usage.percent"
VM_MEM_METRIC = "memory.used.percent"
POD_CPU_METRIC = "cpu.percent"
POD_MEM_METRIC = "memory.percent"
TYPE_COL_NAME = "type"
# 物理机配置
CPU_COUNT = 4   # 4核
MEM_CAP = 8     # 8G
# 整体归一化范围
MINMAX_RANGE = [[0, 0, 0, 0, 0, 0], [100 * CPU_COUNT, 100 * MEM_CAP, 1e9, 1e9, 1e9, 1e9]]
# 模型名称和对应路径
MODEL_PATH = {"regtree": ("pickle", "model/decompose/regtree-printWb.pkl"), "xgboost": ("pickle", "model/decompose/xgboost.pkl"),
              "direct": ("pickle", "model/rf-pod.pkl")}
MODEL_DICT = {}

class PowerEvaluateService(powerevaluate_pb2_grpc.PowerEvaluateServiceServicer):

    def PowerEvaluate(self, request, context):
        # # 测试请求内容
        # print("host: %s, hostType: %s, start: %s, end: %s, algorithm: %s" %
        #       (request.host, request.hostType, request.start, request.end, request.algorithm))
        # return powerevaluate_pb2.PowerEvaluateReply(power='123')

        # 取出预加载模型
        model = MODEL_DICT[request.algorithm]

        # Redis 连接配置
        redis_pool = redis.ConnectionPool(host=REDIS_ADDR, port=REDIS_PORT, password=REDIS_PWD, db=REDIS_DB)
        # 建立连接
        r = redis.Redis(connection_pool=redis_pool)

        # 如果是直接利用 pod 状态进行 pod 能耗评估的方法
        if request.hostType == "pod" and request.algorithm == "direct":
            # 直接获取该 pod 的相关状态信息
            redis_req_list = redis_utils.redis_ts_split(request.start, request.end, request.host)
            pod_df = preprocess.req2df(redis_req_list, r, request.start, request.end)
            ts_list = pod_df.index.values
            pod_df = preprocess.fillna_decompose(pod_df, methods="interpolate")
            pod_df = preprocess.fillna_decompose(pod_df, methods="fb")
            pod_df = preprocess.fillna_decompose(pod_df)
            # print(pod_df.columns)
            pod_cols = list(map(lambda x: "docker." + request.host + "." + x, POD_COLS))
            print(pod_cols)
            pod_power_list = rf_test(pod_df.loc[:, pod_cols].values, model=MODEL_DICT["direct"])
            return powerevaluate_pb2.PowerEvaluateReply(power='%s' % (dict(zip(ts_list, pod_power_list))))

        # 如果是 pod 能耗分解，需要先找到容器所在的物理机
        if request.hostType == "pod":
            # 获取 pod 对应的 vm id
            print("hostOf" + request.host)
            vm_id_podin = redis_utils.one2one_get(r, "hostOf" + request.host)
            print(vm_id_podin)
            # 获取 vm 对应的 物理机 id
            server_id = redis_utils.one2one_get(r, "hostOf" + vm_id_podin)
            print(server_id)
        else:   # 否则的话，传入的 host 即为物理机名
            server_id = request.host
            # print("VM or Hardware")

        # 根据时间戳判断在哪个时间段内去计算有关的物理机请求有哪些
        redis_req_list = redis_utils.redis_ts_split(request.start, request.end, server_id)
        print(redis_req_list)

        # 请求虚拟机目标时间戳下的 dict
        vm_list = preprocess.req_vmlist(redis_req_list, r, request.start, request.end)
        print(vm_list)
        vm_df_dict = dict()
        # 得到每个虚拟机的 dataframe 数据
        for vm_name in vm_list:
            vm_df_dict[vm_name] = preprocess.req2df(redis_utils.redis_ts_split(request.start, request.end, vm_name),
                                                    r, request.start, request.end)
            # 先填充一下虚拟机类型（有该列时删除）
            vm_df_dict[vm_name][TYPE_COL_NAME] = '2'
        # 显示 dataframe 的所有列
        pd.set_option('display.max_columns', None)
        print(vm_df_dict)
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
                                                               cpu_column=None)
                                                               # cpu_column=VM_CPU_METRIC)

        # 如果是虚拟机能耗分解
        if request.hostType == "vm":
            if request.algorithm == "regtree":
                multi = True if request.start != request.end else False
                # # 加载模型文件（已采用预加载模型）
                # file = open("model/decompose/regtree-printWb.pkl", 'rb')
                # model = pickle.load(file)
                # file.close()
                # 从列中取出虚拟机类型
                vm_types = list(map(lambda x: x.reset_index()[TYPE_COL_NAME].values[0], vm_df_dict.values()))
                # 对虚拟机列名重排序后转为 ndarray
                vm_np_list = list(map(lambda x: x.loc[:, MINMAX_COLS].values, vm_df_dict.values()))
                # 取得每个虚拟机 CPU 使用比例
                vm_cpu_usage_dict = cpu.decompose(vm_df_dict, VM_CPU_METRIC, pdu_df, only_cpu=True)
                # 所有虚拟机的CPU使用率之和
                cpu_total = sum(vm_cpu_usage_dict.values())
                # 计算CPU使用率的比例
                vm_cpu_ratio_list = list(map(lambda x: x / cpu_total, vm_cpu_usage_dict.values()))
                # 计算得到各个虚拟机的能耗
                vm_power_list = regtree.predict_wrapper(model, vm_train_df.values, vm_np_list, vm_types, vm_cpu_ratio_list,
                                                        len(MINMAX_COLS), multi=multi)
                # 将功率数据存放到 dict 中
                vm_power_dict = dict(zip(vm_df_dict.keys(), vm_power_list))
                # 将物理机能耗也保存到 dict 中
                vm_power_dict[request.host] = np.squeeze(pdu_df.values).tolist() if multi else pdu_df.values[0].tolist()
                return powerevaluate_pb2.PowerEvaluateReply(power='%s' % (vm_power_dict))

        # pod 能耗分解
        elif request.hostType == "pod":
            if request.algorithm == "regtree":
                # # 加载模型文件
                # with open("model/decompose/regtree-printWb.pkl", 'rb') as file:
                #     model = pickle.load(file)
                # 从 redis 中获取 pod 相关数据
                pod_df = preprocess.req2df(redis_utils.redis_ts_split(request.start, request.end, request.host),
                                           r, request.start, request.end)
                # 取出所有时间戳
                ts_list = pod_df.index.values
                # print(pod_df)
                # 预处理，容器特征名去容器名化
                pod_df = preprocess.rmname_from_cols(pod_df, "docker." + request.host + '.')
                # print(pod_df)
                # 预处理，缺失值处理
                pod_df = preprocess.fillna_decompose(pod_df, methods="interpolate")
                pod_df = preprocess.fillna_decompose(pod_df, methods="fb")
                # print(pod_df)
                # 根据该容器所在虚拟机去确定虚拟机类型，由于此处固定为2，因此写死
                pod_df[TYPE_COL_NAME] = '2'
                # # 预处理，值放大+归一化
                pod_df = preprocess.vm_type_enlarge(pod_df.set_index(TYPE_COL_NAME), VM_CONF, cpu_metric=None, mem_metric=POD_MEM_METRIC,
                                                    minmax_columns=POD_COLS, minmax_range=MINMAX_RANGE)
                print(pod_df)
                # 取得每个虚拟机 CPU 使用比例
                vm_cpu_usage_dict = cpu.decompose(vm_df_dict, VM_CPU_METRIC, pdu_df, only_cpu=True)
                # 所有虚拟机的CPU使用率之和
                cpu_total = sum(vm_cpu_usage_dict.values())
                # print(cpu_total)
                # 计算该 pod CPU 使用率占总CPU使用率的比率
                pod_cpu_ratios = pod_df[POD_CPU_METRIC].values / cpu_total
                # print(pod_cpu_ratios)
                # 将 pod 的列名按照虚拟机模型的顺序排序后取出
                pod_np = pod_df.loc[:, POD_COLS].values
                # 采用能耗分解模型进行评估
                pod_power_list = regtree.predict_wrapper(model, vm_train_df.values, pod_np, pod_df.index.values, pod_cpu_ratios,
                                          len(POD_COLS), multi=True)
                # print(pod_power_list)
                # 将结果用 dict 封装起来
                return powerevaluate_pb2.PowerEvaluateReply(power='%s' % (dict(zip(ts_list, pod_power_list))))

        # 吉万鹏硬件分解模块
        elif request.hostType == "hardware":
            # # 以 iloc 去取的第一行数据
            # timestampi = 0
            # 多点分解逻辑
            power_dict = {"other_power": [], "CPU_power": [], "Disk_power": [], "Memory_power": [], "Net_power": []}
            for timestampi in range(vm_train_df.shape[0]):
                # 取得原始数据模型的预测值
                pre = hardware.xgboost_test(vm_train_df.iloc[timestampi: timestampi + 1], model)
                # 通过修改赋值的方法去做分解
                hardware.hardware_power(vm_train_df, pre, timestampi, feature_columns=vm_train_df.columns,
                                                     power_dict=power_dict, model=model)
                # 加上物理机能耗数据
            print(pdu_df.shape)
            power_dict["host"] = np.squeeze(pdu_df.values).tolist() if pdu_df.shape[0] > 1 else [np.squeeze(pdu_df.values).tolist()]
            return powerevaluate_pb2.PowerEvaluateReply(power='%s' % (power_dict))
        # return powerevaluate_pb2.PowerEvaluateReply(power='%s' % (123))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    powerevaluate_pb2_grpc.add_PowerEvaluateServiceServicer_to_server(PowerEvaluateService(), server)
    print(REDIS_ADDR, REDIS_PORT, REDIS_DB, REDIS_PWD)
    for model in MODEL_PATH.keys():
        path = MODEL_PATH[model][1]
        print(model, path)
        if MODEL_PATH[model][0] == "pickle":
            with open(path, "rb") as fa:
                MODEL_DICT[model] = pickle.load(fa)
        else:
            MODEL_DICT[model] = torch.load(path)
    server.add_insecure_port('[::]:50052')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
