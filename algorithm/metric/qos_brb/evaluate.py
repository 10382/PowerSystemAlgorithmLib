from numpy import *
import numpy as np
import pandas as pd
from . import fun_opt as fo
from . import fun_caloutput as fc


# 使用类创建结构体
class Paramodal(object):
    class Struct(object):
        def __init__(self, rulenum, attrinum, conseqnum, attrixnum):
            self.rulenum = rulenum
            self.attrinum = attrinum
            self.conseqnum = conseqnum
            self.attrixnum = attrixnum

    def make_struct(self, rulenum, attrinum, conseqnum, attrixnum):
        self.rulenum = rulenum
        self.attrinum = attrinum
        self.conseqnum = conseqnum
        self.attrixnum = attrixnum
        return self.Struct(rulenum, attrinum, conseqnum, attrixnum)


class TrainDataSet(object):
    class Struct(object):
        def __init__(self, input, output):
            self.input = input
            self.output = output

    def make_struct(self, input, output):
        self.input = input
        self.output = output
        return self.Struct(input, output)


# 初始化类数组
paramodal = [Paramodal() for i in range(0, 5)]
# print(paramodal)
trainDataSet = TrainDataSet()

for i in range(0, 5):
    attrixnum = []
    attrixnum.append(i)
    paramodal[i].make_struct(i, i, i, attrixnum)


def f_opt(x):
    f = fo.fun_opt(x, trainDataSet)
    return x


def main_opt_bayes(input, start, end, data_path=None):
    # 初始参数
    # 参数的排列是[规则权重，置信度1，置信度2，置信度3（置信结构，初始）]47维数组
    # 47个参数（分号表示列向量）分别是第一个先行属性的三个参考值，第二个先行属性的三个参考值，结果的三个参考值，第一个属性权重，第二个属性权重
    # 浩哥改过的
    x0_m1 = [
        1, 1, 0, 0,
        1.0, 0.86, 0.14, 0,
        1.0, 0.56, 0, 0.44,
        1.0, 0.75, 0.25, 0,
        1.0, 0, 1, 0,
        1.0, 0, 0.4, 0.6,
        1.0, 0.64, 0, 0.36,
        1.0, 0, 0.47, 0.53,
        1.0, 0, 0, 1,
        0.5, 1.5, 2.5, 0,
        400, 750, 0, 50,
        100, 0.571, 0.429]
    # 123个参数
    x0_m2 = [
        1.0, 1, 0, 0,
        1.0, 0.75, 0.2, 0.05,
        1.0, 0.5, 0.25, 0.25,
        1.0, 0.75, 0.2, 0.05,
        1.0, 0.55, 0.2, 0.25,
        1.0, 0.35, 0.5, 0.15,
        1.0, 0.35, 0.5, 0.15,
        1.0, 0.35, 0.5, 0.15,
        1.0, 0.2, 0.6, 0.2,
        1.0, 0.2, 0.7, 0.1,
        1.0, 0.3, 0.4, 0.3,
        1.0, 0.4, 0.45, 0.15,
        1.0, 0.2, 0.25, 0.55,
        1.0, 0.1, 0.15, 0.75,
        1.0, 0.3, 0.4, 0.3,
        1.0, 0.3, 0.4, 0.3,
        1.0, 0.9, 0.1, 0,
        1.0, 0, 0.1, 0.9,
        1.0, 0.1, 0.4, 0.5,
        1.0, 0, 0.4, 0.6,
        1.0, 0, 0.2, 0.8,
        1.0, 0.25, 0.25, 0.5,
        1.0, 0.2, 0.35, 0.45,
        1.0, 0.3, 0.45, 0.25,
        1.0, 0.3, 0.45, 0.25,
        1.0, 1, 0, 0,
        1.0, 1, 0, 0,
        0, 1, 2, 0,
        150, 300, 0, 1,
        2, 0, 50, 100,
        0.118, 0.291, 0.591]
    # 123个参数
    x0_m3 = [
        1.0, 1, 0, 0,
        1.0, 0.75, 0.2, 0.05,
        1.0, 0.5, 0.25, 0.25,
        1.0, 0.75, 0.2, 0.05,
        1.0, 0.55, 0.2, 0.25,
        1.0, 0.35, 0.5, 0.15,
        1.0, 0.35, 0.5, 0.15,
        1.0, 0.35, 0.5, 0.15,
        1.0, 0.2, 0.6, 0.2,
        1.0, 0.2, 0.7, 0.1,
        1.0, 0.3, 0.4, 0.3,
        1.0, 0.4, 0.45, 0.15,
        1.0, 0.2, 0.25, 0.55,
        1.0, 0.1, 0.15, 0.75,
        1.0, 0.3, 0.4, 0.3,
        1.0, 0.3, 0.4, 0.3,
        1.0, 0.9, 0.1, 0,
        1.0, 0, 0.1, 0.9,
        1.0, 0.1, 0.4, 0.5,
        1.0, 0, 0.4, 0.6,
        1.0, 0, 0.2, 0.8,
        1.0, 0.25, 0.25, 0.5,
        1.0, 0.2, 0.35, 0.45,
        1.0, 0.3, 0.45, 0.25,
        1.0, 0.3, 0.45, 0.25,
        1.0, 1, 0, 0,
        1.0, 1, 0, 0,
        0, 500, 1000, 0,
        0.5, 1, 30, 65,
        100, 0, 50, 100,
        0.138, 0.241, 0.621]
    # 123个参数
    # 这些初始值的设定不太懂
    x0_m4 = [
        1.0, 0.9, 0.1, 0,
        1.0, 0.85, 0.15, 0,
        1.0, 0.79, 0.21, 0,
        1.0, 0.38, 0.4, 0.22,
        1.0, 0.04, 0.96, 0,
        1.0, 0.67, 0.33, 0,
        1.0, 0, 0.65, 0.35,
        1.0, 0, 0.89, 0.11,
        1.0, 0.48, 0.52, 0,
        1.0, 0, 0.78, 0.22,
        1.0, 0.04, 0.96, 0,
        1.0, 0.68, 0.32, 0,
        1.0, 0, 0.70, 0.3,
        1.0, 0, 0.94, 0.06,
        1.0, 0.56, 0.44, 0,
        1.0, 0, 0.57, 0.43,
        1.0, 0, 0.81, 0.19,
        1.0, 0.36, 0.64, 0,
        1.0, 0, 0.21, 0.79,
        1.0, 0, 0.45, 0.55,
        1.0, 0, 0.87, 0.13,
        1.0, 0, 0.13, 0.87,
        1.0, 0, 0.37, 0.63,
        1.0, 0, 0.79, 0.21,
        1.0, 0, 0, 1,
        1.0, 0, 0.24, 0.76,
        1.0, 0, 0.66, 0.34,
        0, 50, 100, 0,
        50, 100, 0, 50,
        100, 0, 50, 100,
        0.075, 0.332, 0.593]
    x0 = []
    x0 = x0_m1 + x0_m2 + x0_m3 + x0_m4
    # print(len(x0))

    # 初始化类数组
    paramodal = [Paramodal() for i in range(4)]

    # 参数模型，要根据参数模型做必要的约束
    attrixnum = []
    attrixnum.append(3)
    attrixnum.append(3)
    paramodal[0].make_struct(9, 2, 3, attrixnum)
    attrixnum = []
    attrixnum.append(3)
    attrixnum.append(3)
    attrixnum.append(3)
    paramodal[1].make_struct(27, 3, 3, attrixnum)
    attrixnum = []
    attrixnum.append(3)
    attrixnum.append(3)
    attrixnum.append(3)
    paramodal[2].make_struct(27, 3, 3, attrixnum)
    attrixnum = []
    attrixnum.append(3)
    attrixnum.append(3)
    attrixnum.append(3)
    paramodal[3].make_struct(27, 3, 3, attrixnum)

    k = 0
    # （返回向量x0长度，总的参数数目）
    totalParaNum = len(x0)
    # 生成A矩阵
    # 217 * 20维矩阵，用于等式约束，第一个减第二个，第二个减第三个要小于0，A * x <= b，三个参考值要依次从小到大，不能说第二个参考值比第一个还小，如果结果参考值也调整的话也要满足这个约束
    # 8个输入，4个模型，其中1个是47，其余3个是123，故生成A矩阵
    # 416 * 32维矩阵
    rowNum = []
    paraNum = []
    paraModalSart = []
    A = ndarray((totalParaNum, 30))  # 最终的维度为416*30
    # print("len(paramodal):", len(paramodal))
    for l in range(len(paramodal)):
        # 行数比结构体中的结果数多1
        # print("l:", l)
        rowNum.append(1 + paramodal[l].conseqnum)
        # print("rowNum:", rowNum) # 固定的 4
        # 计算每个模型参数的数目
        # print("paramodal[l].rulenum:", paramodal[l].rulenum)
        paraNum.append(paramodal[l].rulenum * rowNum[l] + sum(paramodal[l].attrixnum) + paramodal[l].conseqnum +
                       paramodal[l].attrinum)
        if l == 0:
            paraModalSart.append(0),
        else:
            paraModalSart.append(sum(paraNum[:-1]))
        for i in range(paramodal[l].attrinum):
            for j in range(paramodal[l].attrixnum[i] - 1):
                # 1个totalparamodal行1列的零矩阵
                A[:, k] = zeros(totalParaNum)
                # print("k", k)
                if i == 0:
                    # print("left", paraModalSart[l] + rowNum[l] * paramodal[l].rulenum + j)
                    A[paraModalSart[l] + rowNum[l] * paramodal[l].rulenum + j, k] = 1
                    A[paraModalSart[l] + rowNum[l] * paramodal[l].rulenum + j + 1, k] = -1
                else:
                    # print("left", paraModalSart[l] + rowNum[l] * paramodal[l].rulenum + sum(paramodal[l].attrixnum[:i]) + j)
                    A[paraModalSart[l] + rowNum[l] * paramodal[l].rulenum + sum(paramodal[l].attrixnum[:i]) + j, k] = 1
                    A[paraModalSart[l] + rowNum[l] * paramodal[l].rulenum + sum(
                        paramodal[l].attrixnum[:i]) + j + 1, k] = -1
                k = k + 1
        # print("K", k)
        for i in range(paramodal[l].conseqnum - 1):
            A[:, k] = zeros(totalParaNum)
            # print(paraModalSart[l] + rowNum[l] * paramodal[l].rulenum + sum(paramodal[l].attrixnum) + i)
            A[paraModalSart[l] + rowNum[l] * paramodal[l].rulenum + sum(paramodal[l].attrixnum) + i, k] = 1
            A[paraModalSart[l] + rowNum[l] * paramodal[l].rulenum + sum(paramodal[l].attrixnum) + i + 1, k] = -1
            k = k + 1

    # optimset没有搬过来，因为源码中后面都没有用到optimset数据，应该是不需要

    numberOfSample = 300

    if data_path != None:
        input = pd.read_excel(data_path, header=None).iloc[start:end, :]
    # print(input.shape)

    # input = array()
    # # 先不要这种实际输出
    # # output = y4
    # # output = [12 15 35 65 26 34 53 28 46 36 29 55 26 66 33 46 63 49 58 27]
    # print(pd.read_excel('start.xlsx', sheet_name='Sheet1')).iloc[1:500, 8]
    # output = pd.read_excel('start.xlsx', sheet_name='Sheet1').iloc[1:500, 8]   # 究竟是

    output = np.ndarray(len(input))
    input = input.T.values
    print(input.shape)
    trainDataSet.make_struct(input, output)
    # trainDataSet.output=y3
    x_opt = x0

    OptResult = []
    IniResult = []
    for i in range(end-start):
        # 计算y3在参数优化后的数值结果（我的注释）
        OptResult.append(fc.fun_caloutput(x_opt, input[:, i]))
        # # 计算y3在参数初始时的数值结果（我的注释）
        # # 由于并没有进行优化，因此初始参数就是优化后的参数
        # IniResult.append(fc.fun_caloutput(x0, input[:, i]))

    return OptResult

