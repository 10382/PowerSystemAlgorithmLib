# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Description entropy.py
# @version 1.1 2021/2/9 22:13
import numpy as np
def entropy(x,ind):
# %实现用熵值法求各指标(列）的权重及各数据行的得分
# %x为原始数据矩阵, 一行代表一个样本, 每列对应一个指标
# %ind指示向量，指示各列正向指标还是负向指标，1表示正向指标，2表示负向指标
# %s返回各行（样本）得分，w返回各列权重
	n,m=x.shape # % n个样本, m个指标
	#%%数据的归一化处理
	X = np.zeros((n,m))
	for i in range(m):
		if ind[i]==1: #%正向指标归一化
			X[:,i]=normalization(x[:,i],1,0.002,0.996)#    %若归一化到[0,1], 0会出问题
		else:# %负向指标归一化
			X[:,i]=normalization(x[:,i],2,0.002,0.996)
	Y = X
	#%%计算第j个指标下，第i个样本占该指标的比重p(i,j)
	p = np.zeros((n,m))
	for i in range(n):
		for j in range(m):
			p[i,j]=X[i,j]/sum(X[:,j])

	#%%计算第j个指标的熵值e(j)
	k=1/np.log(n)
	e = np.zeros((1,m))
	for j in range(m):
		e[0,j]=-k*sum(p[:,j]*np.log(p[:,j]))
	d=np.ones((1,m))-e #%计算信息熵冗余度
	w= d/np.sum(d) #%求权值w
	#%s=100*w*p'; %求综合得分
	return w

def normalization(x,type,ymin,ymax):
	# % 实现正向或负向指标归一化，返回归一化后的数据矩阵
	# % x为原始数据矩阵, 一行代表一个样本, 每列对应一个指标
	# % type设定正向指标1, 负向指标2
	# % ymin, ymax为归一化的区间端点
	x = x.reshape(-1,1)
	n,m = x.shape
	y = np.zeros((n,m))
	xmin = np.min(x)
	xmax = np.max(x)
	if type == 1:
		for j in range(m):
			y[:, j]=((ymax - ymin) * (x[:, j] - xmin) / (xmax - xmin) + ymin)
	else:
		for j in range(m):
			y[:, j]=((ymax - ymin) * (xmax- x[:, j]) / (xmax - xmin) + ymin)
	return y.reshape(-1)