import numpy as np
import math as mt

from . import fun_InfoTran as fi
from . import fun_bab_base as fbb
from . import fun_brb_numRescal as fbn

class myClass:
    def __init__(self):
        self.rulenum = 0
        self.attrinum = 0
        self.conseqnum = 0
        self.attrixnum = []

def fun_opt(x, trainData):
    
    inputNum = trainData.input

    output = trainData.output
    
    dataNum = len(output)

    paramodal = myClass()

    outputPre = []

    y1_relDegree = np.empty([dataNum][dataNum], dtype=float)
    y2_relDegree = np.empty([dataNum][dataNum], dtype=float)
    y3_relDegree = np.empty([dataNum][dataNum], dtype=float)
    y4_relDegree = np.empty([dataNum][dataNum], dtype=float)

    for i in range(dataNum):
        
        x_m1 = x[0: 47]
        x_m2 = x[47: 170]
        x_m3 = x[170: 293]
        x_m4 = x[293: 416]
        
        paramodal.rulenum = 9
        paramodal.attrinum = 2
        paramodal.conseqnum = 3
        paramodal.attrixnum.append(3)
        paramodal.attrixnum.append(3)

        numVal = [[1, inputNum[0][i]], [2, inputNum[1][i]]]

        a1 = fi.fun_InfoTran(x_m1, paramodal, [], numVal)
        y1_relDegree[i, :] = fbb.fun_brb_base(x_m1, paramodal, a1)

        paramodal.rulenum = 27
        paramodal.attrinum = 3
        paramodal.conseqnum = 3
        paramodal.attrixnum.append(3)

        numVal = [[1, inputNum[2][i]] , [2, inputNum[3][i]] , [2, inputNum[4][i]]]

        a2 = fi.fun_InfoTran(x_m2, paramodal, [], numVal)
        y2_relDegree[i, :] = fbb.fun_brb_base(x_m2, paramodal, a2)


        numVal = [[1,input(5,i)], [2,input(6,i)], [3,input(7,i)]]

        a3 = fi.fun_InfoTran(x_m3, paramodal, [], numVal)
        y3_relDegree[i,:] = fbb.fun_brb_base(x_m3, paramodal, a3)


        matchVal=[[1,y1_relDegree[i,:]], [2,y2_relDegree[i,:]], [3,y3_relDegree[i,:]]]

        a4 = fi.fun_InfoTran(x_m4, paramodal, matchVal,[])
        y4_relDegree[i,:] = fbb.fun_brb_base(x_m4, paramodal, a4)
        y4_pred = fbn.fun_brb_numRescal(x_m4, paramodal, y4_relDegree[i,:])

        outputPre.append(y4_pred)

    sum = 0

    for i in range(dataNum):
        sum += mt.pow( outputPre[i] - output[i], 2)

    try:
        f = sum / dataNum
        f = np.real(f)

    except ValueError as identifier:
        print("the f value exist some error:", identifier)

    return f