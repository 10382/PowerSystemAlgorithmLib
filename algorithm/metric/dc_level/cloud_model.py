# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Description cloud_model.py
# @version 1.1 2021/2/9 19:08
import numpy as np

def cloud_model(x):
# % 根据云模型计算输出的y值
	level_num = 5
	indi_num = 5
	y = np.zeros((indi_num, level_num))
	for i in range(indi_num):
		y[i,:] = normal_cloud_c(x[i], i)
	return y

def normal_cloud_c(x,index):
	y = []
	#% % % % % % % % % % % % C1和其他指标参数初始化 % % % % %
	#% C指标一共有5个等级水平
	level_num = 5;
	#% 每个指标不同等级的上下限
	C1_low = np.array([0.0000,0.0600 ,0.1200, 0.1800, 0.2400])
	C1_high = np.array([0.0600,0.1200,0.1800,0.2400,0.3000])

	C2_low = np.array([0.0000,0.2000,0.4000,0.6000,0.8000])
	C2_high = np.array([0.2000,0.4000,0.6000,0.8000,1.0000])

	C3_low = np.array([0.0200,0.0230,0.0260,0.0290,0.0320])
	C3_high = np.array([0.0230,0.0260,0.0290,0.0320,0.0350])

	C4_low = np.array([0.0000,0.2000 ,0.4000 ,0.7000 ,0.8000])
	C4_high = np.array([0.2000, 0.4000,0.7000 ,0.8000,1.0000])

	C5_low = np.array([1.0000,1.0700,1.1400,1.2100,1.2800])
	C5_high = np.array([1.0700,1.1400,1.2100,1.2800,1.3500])

	C_low = [C1_low, C2_low, C3_low, C4_low, C5_low]
	C_high = [C1_high, C2_high, C3_high, C4_high, C5_high]
	Cx_low = C_low[index]
	Cx_high = C_high[index]
	# % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

	# % % % % % % % % % % % 计算云模型 % % % % % % % % % % % % % % % % % %
	# % 计算Ex
	Ex = np.zeros((1, level_num))
	for i in range(level_num):
		Ex[0,i] = (Cx_low[i] + Cx_high[i]) / 2

	#% 计算En
	En = np.zeros((1, level_num))
	for i in range(level_num):
		En[0,i] = (Cx_high[i] - Cx_low[i]) / 2.355

	# % 计算He
	He = 0.01

	#% n = 2000;
	#% 正态分布Enn
	for i in range(level_num):
		Enn = np.random.normal(En[0,i], He)
		#%   x = normrnd(Ex(i), Enn);
		y.append(np.exp((-(x - Ex[0,i])**2)/ (2 * Enn**2)))
	return np.array(y)

def score_cal(Y):
	#计算得分值r
	score = [1,2,3,4,5]
	res = 0
	for i in range(5):
		res =res + Y[i]*score[i]
	r = res/sum(Y)
	return r



