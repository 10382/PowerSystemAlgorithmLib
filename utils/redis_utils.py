# -*- coding: UTF-8 -*-

from __future__ import print_function
import sys
import json
import redis
# import rediscluster as rc

REDIS_NODES = [{'host':'192.168.1.23','port':7133},
               {'host':'192.168.1.24','port':8134},
               {'host':'192.168.1.23','port':8133},
               {'host':'192.168.1.24','port':7134},
               {'host':'192.168.1.25','port':8135},
               {'host':'192.168.1.25','port':7135}]

# def redis_cluster():
#     try:
#         redis_conn = rc.StrictRedisCluster(startup_nodes=REDIS_NODES)
#     except Exception as e:
#         print("Connect Error!\n", e)
#         sys.exit(1)
#     redis_conn.set("")

# 将redis中的hash数据取出并转为dict
def redis2dict(in_dict):
    # print(dict(zip(map(bytes.decode, in_dict.keys()), map(json.loads, in_dict.values()))))
    # print(list(map(bytes.decode, in_dict.keys())))
    # print(list(map(json.loads, in_dict.values())))
    return dict(zip(map(bytes.decode, in_dict.keys()), map(json.loads, in_dict.values())))

# 将redis中的set数据取出并转为list
def redis_set2list(in_set):
    print(in_set)
    return list(map(bytes.decode, in_set))

# @profile
# redis 按数据
def redis_ts_split(start, end, hostname, interval=100):
    # print(type(start))
    start = int(start)
    end = int(end)
    new_start = (start // interval) * interval
    new_end = (end // interval + 1) * interval - 1
    req_list = []
    for ts_start in range(new_start // interval, (new_end + 1) // interval):
        req_list.append("%s.%d.%d" % (hostname, ts_start * interval, (ts_start * interval + interval - 1)))
    # print(req_list)
    return req_list

# 用于读取 redis 中一对一的映射关系(比如 pod 到虚拟机，虚拟机到物理机)，返回得到的value
def one2one_get(redis_conn, redis_key):
    return bytes.decode(redis_conn.get(redis_key))


