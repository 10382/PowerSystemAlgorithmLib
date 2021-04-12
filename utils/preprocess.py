# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import chain
from operator import itemgetter
from utils import redis_utils
from sklearn.preprocessing import MinMaxScaler

# @profile
# 根据请求的 key 值列表去数据库取数据拼接，并做切片，返回DataFrame
def req2df(req_list, r, start=None, end=None):
    for i, req in enumerate(req_list):
        get_dict = r.hgetall(req)
        get_dict = redis_utils.redis2dict(get_dict)
        if i == 0:
            if start == end:
                df = pd.DataFrame(get_dict, index=[start])
            else:
                df = pd.DataFrame(get_dict)
        else:
            df = pd.concat([df, pd.DataFrame(get_dict)])
        if start != None and end != None:
            df = df.loc[start: end, :]
    return df

# 虚拟机分解模块需要用到的redis中key值为单时间戳取数据的逻辑
def req_vmlist(req_list, r, startTs, endTs=None):
    get_dict = dict()
    # ？？？每次把内部先转换成list后再拼接快一点，还是先拼接，然后最后再转换快一点？？？
    for req in req_list:
        print(req)
        tmp_dict = dict()
        tmp_dict = {**tmp_dict, **r.hgetall(req + ".virtualMachineSet")}
        # print(tmp_dict)
        get_dict.update(redis_utils.redis2dict(tmp_dict))
        # print("req_vmlist(): get_dict\n", get_dict)
    if endTs == None or startTs == endTs:
        vm_list = get_dict[str(startTs)]
    else:
        # print(list(map(str, range(int(startTs), int(endTs)))))
        # 将所有时间戳内的虚拟机列表取出来，再通过chain合并为单个list，通过set去重，最后再转回list
        vm_list = list(set(chain(*get_dict.values())))
        print("req_vmlist(): vm_list\n", vm_list)
        # print(vm_list)
    return vm_list

    # req_vm_list = list(map(lambda x: x+'.'+targetTs, vm_list))
    # vm_df_list = []
    # for req_vm in req_vm_list:
    #     vm_df_list.append(req2df([req_vm], r, targetTs, targetTs))
    # return vm_df_list

# 将多个虚拟机的 dataframe 合并为单个关于虚拟机类型的 dataframe
def vm_type_sum(df_list, type_dict, type_col_name="type", multi_gb_names=None):
    vm_df = pd.concat(df_list, axis=0)
    # print(vm_df)
    # # 显示所有列
    # pd.set_option('display.max_columns', None)
    # 根据 type 去聚合得到每个虚拟机类型下的特征和
    if  multi_gb_names == None:
        vm_sum_df = vm_df.groupby([type_col_name]).sum()
    else:
        vm_sum_df = vm_df.reset_index().groupby(multi_gb_names).sum()
        for name in multi_gb_names:
            if name != type_col_name:
                idx_remove = name
                break
        vm_sum_df.reset_index(level=idx_remove, inplace=True)
        # print(vm_sum_df)
    for type in type_dict.keys():
        if type not in vm_sum_df.index:
            vm_sum_df.loc[type] = 0
    print("VM sum dataframe:\n", vm_sum_df)
    return vm_sum_df.sort_index()

# 根据虚拟机类型去做特征放大（比如说不同的配置的内存，使用率都是100%，但实际用量是不同的）并归一化
def vm_type_enlarge(vm_df, vm_conf, cpu_metric=None, mem_metric=None,
                    minmax_columns=None, minmax_range=None):
    if cpu_metric == None and mem_metric == None:
        return vm_df
    # print(vm_df)
    for idx in vm_conf.keys():
        # print(vm_conf[idx])
        if idx not in vm_df.index.values:
            continue
        for col in vm_conf[idx].keys():
            # 如果指定了 cpu_metric 和 mem_metric，就代表这两个指标需要扩大。
            if (cpu_metric != None and cpu_metric == col) or (mem_metric != None and mem_metric == col):
                # print(col, '\n', vm_df.loc[idx, col])
                vm_df.loc[idx, col] = vm_df.loc[idx, col] * vm_conf[idx][col]
                # print(vm_df.loc[idx, col])
    if minmax_columns != None:
        for i, col in enumerate(minmax_columns):
            if col in vm_df.columns:
                if minmax_range != None:
                    vm_df.loc[:, col] = (vm_df.loc[:, col] - minmax_range[0][i]) / (minmax_range[1][i] - minmax_range[0][i])
    return vm_df

# 将索引为虚拟机类型的 长dataframe 转换为以时间戳为索引的 宽dataframe，根据 type 从小到大按序排好
def vm_type_concat(vm_df, ts_col_name, type_list, start, end, resort_columns=None, cpu_column=None):
    # print(vm_df)
    cpu_columns = []
    for i, type in enumerate(type_list):
        # print(vm_df.loc[[type], resort_columns])
        # print("Type:", type)
        if resort_columns == None:
            tmp_df = vm_df.copy().loc[[type], :].reset_index(drop=True).set_index(ts_col_name)
            if i == 0:
                vm_wide_df = tmp_df.rename(lambda x: str(type) + '-' + x, axis=1)
            else:
                vm_wide_df = pd.concat((vm_wide_df, tmp_df.rename(lambda x: str(type) + '-' + x, axis=1)), axis=1)
        else:
            tmp_df = vm_df.copy().loc[[type], resort_columns].reset_index(drop=True).set_index(ts_col_name)
            if i == 0:
                vm_wide_df = tmp_df.rename(lambda x: str(type) + '-' + x, axis=1)
            else:
                vm_wide_df = pd.concat((vm_wide_df, tmp_df.rename(lambda x: str(type) + '-' + x, axis=1)), axis=1)
        if cpu_column != None:
            cpu_columns.append(str(type) + '-' + cpu_column)
    # print(vm_wide_df)
    # print(start, end)
    ret_df = vm_wide_df.loc[start:end, :].fillna(0)
    if cpu_column != None:
        return ret_df, ret_df.loc[:, cpu_columns].sum(axis=1)
    else:
        return ret_df, None


# 将单点的时间戳转换为 Redis 中存储的段结构
def ts_select(ts, interval=100):
    if not isinstance(ts, int):
        ts = int(ts)
    tmp = ts // interval
    return tmp * interval, (tmp + 1) * interval - 1

# 虚拟机分解模块需要用到的redis中key值为多时间戳取数据的逻辑
def req_vm2df_multits(req, r, targetTs):
    vm_list = redis_utils.redis_set2list(r.smembers(req + ".virtualMachineSet"))
    ts_start, ts_end = ts_select(targetTs)
    req_vm_list = list(map(lambda x: x+'.'+str(ts_start)+'.'+str(ts_end), vm_list))
    vm_df_list = []
    for req_vm in req_vm_list:
        vm_df_list.append(req2df([req_vm], r, targetTs, targetTs))
    return vm_df_list

# 缺失值填充
def fillna_decompose_old(df, value=None, columns=None, methods="zero"):
    if columns == None:
        columns = df.columns.values
    if value != None:
        df[columns] = df[columns].fillna(value)
    else:
        if methods == "zero":
            df[columns] = df[columns].fillna(0)
        else:
            pass	# 待完工

# @profile
# 缺失值填充，带返回值
def fillna_decompose(df, value=None, columns=None, methods="zero"):
    data_new = df
    if columns == None:
        columns = data_new.columns.values
    if value != None:
        data_new.loc[:,columns] = data_new.loc[:,columns].fillna(value)
    else:
        if methods == "zero":
            data_new.loc[:,columns] = data_new.loc[:,columns].fillna(0)
        elif methods == "fromfront":
            data_new.loc[:,columns] = data_new.loc[:,columns].fillna(method='ffill')
        elif methods == "fromback":
            data_new.loc[:, columns] = data_new.loc[:, columns].fillna(method='bfill')
        elif methods == "fb":
            data_new.loc[:,columns] = data_new.loc[:,columns].fillna(method='bfill')
            data_new.loc[:,columns] = data_new.loc[:,columns].fillna(method='ffill')
        elif methods == "interpolate":
            data_new.loc[:, columns] = data_new.loc[:, columns].interpolate()
    return data_new

# @profile
# 异常值处理
def outlier_decompose(df, columns=None, types="nonega", upbounds=None):
    if columns == None:
        columns = df.columns.values
    df[df[columns] < 0] = 0
    if types == "nonega":
        return
    if isinstance(upbounds, list):
        for i in range(len(columns)):
            df[df[[columns[i]]] > upbounds[i]] = upbounds[i]
    else:
        df[df[columns] > upbounds] = upbounds

# def outlier_np_minmax(data, types="nonega", upbounds=None):


# 百分比数据异常值处理
def correct_percent(data, col_name):
    data_new = data
    len_data = len(data)
    for i in range(1, len_data):
        if data_new.loc[i, col_name]>100:
            data_new.loc[i, col_name] = 100
        elif data_new.loc[i, col_name] < 0:
            data_new.loc[i, col_name] = 0
    return data_new

# 非负数据异常值处理
def correct_statistics(data, col_name):
    data_new = data
    len_data = len(data)
    for i in range(1, len_data):
        if data_new.loc[i, col_name] < 0:
            data_new.loc[i, col_name] = 0
    return data_new

# @profile
# 最大最小归一化
def minmax(data, columns=None, range=None):
    scaler = MinMaxScaler()
    if range is None:
        return scaler.fit_transform(data)
    else:
        scaler.fit(range)
        # scaler.fit(range if isinstance(range, list) else range.tolist())
        return scaler.transform(data.loc[:, columns] if columns else data)

# @profile
# 预测数据格式转换
def pred_concat(data, labels, time_len=10, train=False, overlap=True):
    data_new = data
    for i in range(1, time_len):
        data_new = np.concatenate((data_new, np.roll(data, -1, axis=0)), axis=1)
    if train:
        return data_new[:-(time_len)], labels[time_len:]
    else:
        return data_new[:-(time_len)+1], None

def rmname_from_cols(df, patten_rm):
    return df.rename(lambda x: x.replace(patten_rm, ''), axis="columns")