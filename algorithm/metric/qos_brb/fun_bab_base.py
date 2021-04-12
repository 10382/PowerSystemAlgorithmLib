import numpy as np
import math as mt

class Myclass:
    def __init__(self):
        self.activeweight = 0
        pass

    def set_activeweight(self, aw):
        self.activeweight = aw

    def make_struct(self, weight):
        self.weight = weight

def fun_brb_base(x, paramodal, a):
    "函数结果是输出置信度belief degree，计算置信度结果，公式（4）"

    Rule = [Myclass() for i in range(paramodal.rulenum)]
    # Rule = [Myclass()] * paramodal.rulenum

    onerowNum = 1 + paramodal.conseqnum

    ResultBelief = np.ndarray((paramodal.rulenum, paramodal.conseqnum), dtype=float)
    attriRefVal = np.ndarray((paramodal.attrinum, 3), dtype=float)
    atrriWeight = []
    attriWeightStd = []
    result_belief = []
    tempA = []


    for i in range(paramodal.rulenum):
        # print("x", i*onerowNum + 1, x[i*onerowNum])
        Rule[i].weight = x[i*onerowNum]
        for j in range(paramodal.conseqnum):
            ResultBelief[i][j] = x[i*onerowNum + j + 1]

    refNum = 0
    for i in range(paramodal.attrinum):
        for j in range(paramodal.attrixnum[i]):
            attriRefVal[i][j] = x[paramodal.rulenum * onerowNum + refNum]
            refNum = refNum + 1

    for i in range(paramodal.attrinum):
        atrriWeight.append(x[paramodal.rulenum * onerowNum + paramodal.conseqnum + refNum + i])


    for i in range(paramodal.attrinum):
        attriWeightStd.append(atrriWeight[i] / np.max(atrriWeight))

    sumx = 0
    for l in range(paramodal.rulenum):
        # print("sum+", Rule[l].weight, "x", np.power(a[l, :], attriWeightStd))
        # if Rule[l].weight * np.prod(np.power(a[l, :], attriWeightStd)) == np.nan:
        #     pass
        sumx = sumx + Rule[l].weight * np.prod(np.power(a[l, :], attriWeightStd))
    # sum = np.real(sum)

    for k in range(paramodal.rulenum):
        # print(sumx)
        # Rule[k].activeweight = Rule[k].weight * np.prod(np.power(a[k, :], attriWeightStd)) / sumx
        Rule[k].set_activeweight(Rule[k].weight * np.prod(np.power(a[k, :], attriWeightStd)) / sumx)
        # print("activeweight", k, Rule[k].activeweight)

    # Rule[6].activeweight = 1

    temp=1.0

    # print(Rule[6].activeweight)

    for l in range(paramodal.rulenum):
        # print(l, temp, Rule[l].activeweight)
        temp = temp * (1 - Rule[l].activeweight)

    for l in range(paramodal.conseqnum):
        tempA.append(1.0)
        for t in range(paramodal.rulenum):
            tempA[l] = tempA[l] * (Rule[t].activeweight * ResultBelief[t][l] + 1 - Rule[t].activeweight)

    sumx = 0.0
    
    for l in range(paramodal.conseqnum):
        sumx = sumx + tempA[l]

    for l in range(paramodal.conseqnum):
        result_belief.append((tempA[l] - temp) / (sumx - paramodal.conseqnum * temp))

    return result_belief