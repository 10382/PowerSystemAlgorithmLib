from . import fun_bab_base as fbb
from . import fun_brb_numRescal as fbn
from . import fun_InfoTran as fi

# class Paramodal(object):
#     class Struct(object):
#         def __init__(self, rulenum, attrinum, conseqnum, attrixnum):
#             self.rulenum = rulenum
#             self.attrinum = attrinum
#             self.conseqnum = conseqnum
#             self.attrixnum = attrixnum

#     def makeStruct(self, rulenum, attrinum, conseqnum, attrixnum):
#             return self.Struct(rulenum, attrinum, conseqnum, attrixnum)

class myClass:
    def __init__(self):
        self.rulenum = 0
        self.attrinum = 0
        self.conseqnum = 0
        self.attrixnum = []

def fun_caloutput(x, input):
    "计算y3的数值结果"
    x_m1 = x[:47]
    x_m2 = x[47:170]
    x_m3 = x[170:293]
    x_m4 = x[293:]
    # print("x_m3", x_m3)

    paramodal = myClass()

    # attrixnum = []
    # attrixnum.append(3)
    # attrixnum.append(3)

    # paramodal = myClass.makeStruct(9, 2, 3, attrixnum)

    paramodal.rulenum = 9
    paramodal.attrinum = 2
    paramodal.conseqnum = 3
    paramodal.attrixnum.append(3)
    paramodal.attrixnum.append(3)

    numVal = [[1, input[0]], [2, input[1]]]
    a1 = fi.fun_InfoTran(x_m1, paramodal, [], numVal)
    y1_relDegree = fbb.fun_brb_base(x_m1, paramodal, a1)


    # attrixnum.append(3)

    # paramodal = myClass.makeStruct(27, 3, 3, attrixnum)
    
    paramodal.rulenum = 27
    paramodal.attrinum = 3
    paramodal.conseqnum = 3
    paramodal.attrixnum.append(3)

    numVal = [[1, input[2]], [2, input[3]], [3, input[4]]]
    a2 = fi.fun_InfoTran(x_m2, paramodal, [], numVal)
    y2_relDegree = fbb.fun_brb_base(x_m2, paramodal, a2)

    # paramodal = myClass.makeStruct(27, 3, 3, attrixnum)

    numVal = [[1, input[5]], [2, input[6]], [3, input[7]]]
    a3 = fi.fun_InfoTran(x_m3, paramodal, [], numVal)
    y3_relDegree = fbb.fun_brb_base(x_m3, paramodal, a3)

    # paramodal = myClass.makeStruct(27, 3, 3, attrixnum)

    matchVal = [[1] + y1_relDegree, [2] + y2_relDegree, [3] + y3_relDegree]
    a4 = fi.fun_InfoTran(x_m4, paramodal, matchVal, [])
    y4_relDegree = fbb.fun_brb_base(x_m4, paramodal, a4)
    y4_pred = fbn.fun_brb_numRescal(x_m4, paramodal, y4_relDegree)

    f = y4_pred

    return f