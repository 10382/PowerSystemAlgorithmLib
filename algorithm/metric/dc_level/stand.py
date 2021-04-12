# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Description stand.py
# @version 1.1 2021/2/10 0:48
import numpy as np
def stand(X):
	X = np.array(X)
	nr,nx = X.shape
	X0 = np.zeros((nr,nx))
	for mk in range(nr):
		X0[mk,:] = (X[mk,:]-np.mean(X,axis=0))/np.std(X,axis=0,ddof=1)
	return X0
if __name__ =='__main__':
	print(stand([[0,1,2],[3,4,5]]))
