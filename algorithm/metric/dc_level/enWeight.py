# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Description enWeight.py
# @version 1.1 2021/2/9 22:07
import pandas as pd
import scipy.io as scio
from entropy import *

sample = scio.loadmat('sample.mat')['sample']  #% load the data

X = sample

# %说明指标是正向指标还是负向指标
# %此数据第一个是负向指标， 其余为正向指标
Ind=[2,2,2,2,2] #%Specify the positive or negative direction of each indicator

# %S 为分数排名 W为指标权重
# %[S,W]=shang(X,Ind) % get the score
W=entropy(X,Ind)

# %保存客观权重变量
wo = W;
wo = {"wo":W}
scio.savemat("./wo.mat",wo)

# %加载数据  列数表示指标数 ， 行数表示评价的个体数
# %此数据 7个评价个体 3个评价指标
