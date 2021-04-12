import pandas as pd
import scipy.io as scio
import numpy as np
from .cloud_model import *

def sample_dc_levels(start, end, data_path="algorithm/metric/dc_level/samples.xlsx", weights_main_path="algorithm/metric/dc_level/ws.mat",
					 weights_object_path="algorithm/metric/dc_level/wo.mat"):
	# input_path = "./samples.xlsx"
	data = pd.read_excel(data_path)
	data = data.values
	row, column = data.shape

	#权重的计算#
	#读取主观权重ws和客观权重wo
	ws = scio.loadmat(weights_main_path)['ws'][0]
	wo = scio.loadmat(weights_object_path)['wo'][0]
	P = sorted(ws)

	#计算a和b
	n = 5
	temp = 0
	for i in range(n):
		temp = temp + (i+1)*P[i]
	a = 1/(n-1)*(temp-(n+1)/n)
	b = 1-a
	w = a*ws+b*wo
	# wsum = w
	# wsum = {"wsum":np.array(w)}
	# scio.savemat("./wsum.mat",wsum)
	#################

	#计算指标数据
	#记录每个试验数据每次试验下对应的 Y1 Y2 Y3 Y4 Y5
	Y = []
	levels = []
	#计算实验次数
	N = 100

	np.random.seed(2021)

	for i in range(row):

		r = []
		#用来存储最后处理完成的结果
		tempY = np.zeros((1, n))
		# print('第 %d 个数据为 [%.3f %.3f %.3f %.3f %.3f] \n'%(i+1,data[i,0],data[i,1],data[i,2],data[i,3],data[i,4]))

		for num in range(N):
			y = cloud_model(data[i,:])

			#乘对应权重
			for k in range(5):
				y[k,:] = w[k]*y[k,:]
			Y = sum(y,0)
			r.append(score_cal(Y))
			level = np.argmax(Y)
			tempY = tempY + Y

		#计算Er和Ern
		r = np.array(r)
		Er = sum(r)/len(r)
		Ern = np.sqrt(sum((r-Er)**2)/len(r))
		theta = Ern/Er
		tempY = tempY/100
		# m = np.max(tempY)
		p = np.argmax(tempY)
		levels.append(p + 1)
	if start // 10 == end // 10:
		return levels[start%10:end%10]
	elif (end // 10 - start // 10) <= 1:
		return levels[start%10:] + levels[:end%10]
	else:
		return levels[start%10:] + levels * (end // 10 - start // 10 - 1) + levels[:end%10]


