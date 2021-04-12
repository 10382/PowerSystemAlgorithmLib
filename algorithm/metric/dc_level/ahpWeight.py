# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Description ahpWeight.py
# @version 1.1 2021/2/10 0:24
import numpy as np
import scipy.io as scio
A=np.array([[1,1/2,6,6,6],[2,1,6,6,6],[1/6,1/6,1,1/3,2],[1/6,1/6,3,1,2],[1/6,1/6,1/2,1/2,1]])#%%在方框内输入矩阵


n,n=A.shape
d,v=np.linalg.eig(A)
r=d[0]
CI=(r-n)/(n-1)
RI=[0,0,0.58,0.90,1.12,1.24,1.32,1.41,1.45,1.49,1.52,1.54,1.56,1.58,1.59]
CR=CI/RI[n-1]
if CR<0.10:
	CR_Result='通过'
else:
	CR_Result='不通过'

# %%权向量计算
w=v[:,0]/np.sum(v[:,0])


#%结果输出
print('该判断矩阵权向量计算报告：')
print('一致性指标:',CI)
print('一致性比例:' ,CR)
print('一致性检验结果:',CR_Result)
print('特征值:' ,r)
print('权向量:' ,w)

# %保存每个指标的主观权重
ws = w
ws = {"ws":ws}
scio.savemat("ws.mat",ws)