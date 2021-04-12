import numpy as np

class Myclass:
    pass

def fun_brb_numRescal(x, paramodal, belief):
    "计算数值公式（9）"
    onerowNum = 1 + paramodal.conseqnum

    Rule = [Myclass() for i in range(paramodal.rulenum)]

    ResultBelief = np.empty([paramodal.rulenum, paramodal.conseqnum], dtype=float)

    for i in range(paramodal.rulenum):
        Rule[i].weight = x[i * onerowNum]
        for j in range(paramodal.conseqnum):
            ResultBelief[i][j] = x[i * onerowNum + j + 1]

    refNum = 0

    attriRefVal = np.ndarray((paramodal.attrinum, paramodal.attrixnum[0]), dtype=float)

    for i in range(paramodal.attrinum):
        for j in range(paramodal.attrixnum[i]):
            attriRefVal[i][j] = x[paramodal.rulenum * onerowNum + refNum]
            refNum = refNum + 1

    result = []

    for i in range(paramodal.conseqnum):
        result.append(x[paramodal.rulenum * onerowNum + refNum + i])

    output = 0.0
    for l in range(paramodal.conseqnum):
        output = output + belief[l] * result[l]
    return output