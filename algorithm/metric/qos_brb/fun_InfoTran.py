# 计算属性匹配度
import numpy as np

# 使用类创建结构体
# class rule(object):
#     class Struct(object):
#         def __init__(self, weight):
#             self.weight = weight
#
#     def make_struct(self, weight):
#         self.weight = weight
#         return self.Struct(weight)

class rule(object):
    def __init__(self):
        pass

    def make_struct(self, weight):
        self.weight = weight

# 初始化类数组
Rule = [rule() for i in range(0, 32)]

def fun_InfoTran(x,paramodal,matchVal,numVal):
    # 部分是数值输入，部分是参考值匹配值的输入
    # 输入参数里面有必要说明哪几个是数值输入，哪几个是匹配度的输入
    # 这个函数求出a矩阵（是综合匹配度计算吗）
    # x是模型的所有参数，paramodal描述模型各种参数的数量，matchVal是参考值匹配度输入，numVal是数值输入
    # matchVal的格式[第n个参考值,匹配值,...;第m全参考值,匹配值,...],计算属性匹配度中的一个值，看公式就知道了
    # numVal的格式[第n个参考值,数值;第n个参考值,数值]
    # 先判断输入的参数是否有误
    matchVal=np.array(matchVal)
    numVal=np.array(numVal)
    if matchVal.shape[0]+numVal.shape[0]!=paramodal.attrinum:
        print("input of fun_InofTran is error")
    # 从参数里面取出规则权重和规则结果的置信度，x存储了所有的参数
    # print(x)
    onerowNum = 1 + paramodal.conseqnum
    ResultBelief = np.zeros((paramodal.rulenum, paramodal.conseqnum)) # ？？？没看懂
    for i in range(paramodal.rulenum):
        # Rule[i].weight=x[onerowNum * (i-1)+1,1] # 直接修改为乘
        Rule[i].weight=x[onerowNum * i + 1] # 直接修改为乘
        for j in range(paramodal.conseqnum):
            # ResultBelief[i,j]=x[onerowNum * (i-1)+1+j,1] # 直接修改为乘
            ResultBelief[i,j]=x[onerowNum * i + 1 + j] # 直接修改为乘

    # 获取输入参考值和计算总的输入参考值的数量
    refNum = 0
    # attriRefVal = np.ndarray((paramodal.attrinum, paramodal.attrixnum))
    # matchDgree = np.ndarray((numVal.shape[0], paramodal.attrixnum))
    # inputIndex = np.ndarray((numVal.shape[0], paramodal.attrixnum))
    attriRefVal = np.ndarray((paramodal.attrinum, 3))
    matchDgree = np.ndarray((numVal.shape[0] if len(matchVal) == 0 else matchVal.shape[0], 3))
    inputIndex = np.ndarray((numVal.shape[0]), dtype=int)
    
    for i in range(paramodal.attrinum):
        for j in range(paramodal.attrixnum[i]):
            attriRefVal[i,j]=x[paramodal.rulenum*onerowNum+refNum]
            refNum = refNum + 1

    # 计算数值输入的匹配度
    if numVal.shape[0]!=0:
        for i in range(numVal.shape[0]):
            # 得到第几个属性是数值
            # nthAttr=int(numVal[i,0])
            nthAttr=int(numVal[i,0]) - 1
            for j in range(paramodal.attrixnum[nthAttr]):
                # 先把所有的匹配度置0
                matchDgree[nthAttr,j] = 0
            for j in range(paramodal.attrixnum[nthAttr]-1):
                
                if numVal[i,1]>=attriRefVal[nthAttr,j] and numVal[i,1]<=attriRefVal[nthAttr,j+1]:
                    inputIndex[nthAttr]=j
                    matchDgree[nthAttr, j] = (attriRefVal[nthAttr, inputIndex[nthAttr] + 1] - numVal[i, 1]) / (attriRefVal[nthAttr, inputIndex[nthAttr] + 1] - attriRefVal[nthAttr,inputIndex[nthAttr]])
                    matchDgree[nthAttr, j + 1] = 1 - (attriRefVal[nthAttr, inputIndex[nthAttr] + 1] - numVal[i, 1]) / (attriRefVal[nthAttr, inputIndex[nthAttr] + 1] - attriRefVal[nthAttr,inputIndex[nthAttr]])
                    break
            if numVal[i,1]<attriRefVal[nthAttr,0]:
                matchDgree[nthAttr,0]=1
            if numVal[i,1]>attriRefVal[nthAttr,j+1]:
                matchDgree[nthAttr,j+1]=1
    # 将输入的匹配度合并到一起
    if matchVal.shape[0]!=0:
        for i in range(matchVal.shape[0]):
            # 得到第几个属性是数值
            nthAttr = int(matchVal[i,0]) - 1
            for j in range(paramodal.attrixnum[nthAttr]):
                matchDgree[nthAttr, j] = matchVal[i, j+1]

    # # # 结果参考值
    # for i=1:paramodal.conseqnum;
    # result(i) = x(paramodal.rulenum * onerowNum + refNum + i);
    # end;
    #
    # # # 属性权重
    # for i=1:paramodal.attrinum;
    # atrriWeight(i) = x(paramodal.rulenum * onerowNum + paramodal.conseqnum + refNum + i);
    # end;

    # # # 判断输入位于参考值集合的什么位置，用于输入与规则的匹配
    # for i=1:paramodal.attrinum;
    # for j=1:paramodal.attrixnum(i) - 1;
    # if (input(i) >= attriRefVal(i, j) & & input(i) <= attriRefVal(i, j + 1))
        # inputIndex(i) = j;
    # break;
    # end
    # end
    # # if (inputIndex(i) == 0)
        # # pause;
    # # end
    # end

    # for i=1:length(inputIndex)
    #
    # if (inputIndex(i) == 0) # 这说明有输入超出了模型设定的参考值范围
        # f = inf;
    # fprintf('#d\n', input(i));
    # for j=1:paramodal.attrixnum(i)
    # fprintf('#f ', attriRefVal(i, j));
    # end
    # # fprintf('#d\n', inputIndex(i));
    # return;
    # end
    # end
    #
    # if (length(inputIndex) < paramodal.attrinum) # # 这说明有输入超出了模型设定的参考值范围
        # f = inf;
    # fprintf('#d\n', length(inputIndex));
    # return;
    # end

    # 生成参考每条规则的参考值
    nrule = 1
    ntotal = 1
    for k in range(paramodal.attrinum):
        ntotal=ntotal*paramodal.attrixnum[k]

    # S = []
    for k in range(paramodal.attrinum-1,-1,-1):
        ncurr=1
        r=[]
        for p in range(k+1,paramodal.attrinum):
            # 当前集合中每个元素的重复次数
            ncurr=ncurr*paramodal.attrixnum[p]
        # 当前集合的循环次数
        nrecurr = ntotal / (ncurr * paramodal.attrixnum[k])
        for p in range(int(nrecurr)):
            for h in range(paramodal.attrixnum[k]):
                for g in range(ncurr):
                    r.append([attriRefVal[k,h]])
        # print(r, S)
        S = np.concatenate((r, S), axis=1) if k != paramodal.attrinum-1 else r
    i=S.shape[0]
    j=S.shape[1]
    a=np.ndarray((paramodal.rulenum, paramodal.attrinum))
    for k in range(i):
        Rule[k].x = []
        for l in range(j):
            Rule[k].x.append(S[k,l])

    # 计算a(k,i)
    ruleIndex=[]
    for k in range( paramodal.rulenum):
        # 先判断规则的参考值位于参考值集合的什么位置
        for i in range( paramodal.attrinum):
            for j in range( paramodal.attrixnum[i]):
                # 因为0老是出现-1e-5，所以0和0匹配不上，就把=改成绝对差<0.001
                # print(Rule[k].x[i], attriRefVal[i,j])
                if abs(Rule[k].x[i]-attriRefVal[i,j])<0.001:
                    if len(ruleIndex) == i:
                        ruleIndex.append(j)
                    else:
                        ruleIndex[i] = j
                    break
            # if (isempty(ruleIndex))
                # pause;
            # end
            a[k,i]=matchDgree[i,ruleIndex[i]]
    return a
    # #
    # for i=1:paramodal.attrinum
    # if (ruleIndex(i) == inputIndex(i)) # # 如果输入在参考值集合的位置和规则的参考值在参考值集合的位置一致
        # a(k, i) = (attriRefVal(i, inputIndex(i) + 1) - input(i)) / (attriRefVal(i, inputIndex(i) + 1) - attriRefVal(i, inputIndex(i)));
    # else if (ruleIndex(i) == inputIndex(i) + 1)
        # a(k, i) = 1 - (attriRefVal(i, inputIndex(i) + 1) - input(i)) / (attriRefVal(i, inputIndex(i) + 1) - attriRefVal(i, inputIndex(i)));
    # else
    # a(k, i) = 0;
    # end
    # end
    #
    # end