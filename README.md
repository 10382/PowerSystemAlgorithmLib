# 算法库开发指引

## 如何安装

算法库是通过python环境下的gRPC 服务框架对外提供服务的，因此需要相应的python环境来执行对应的代码。

目前使用的python版本为3.6.6，其他版本的可行性需自己尝试。

以及相关的python模块包可通过 `pip install -r requirements.txt` 进行安装。

## 如何上手

如果只是想将算法库模块的代码运行起来，且在[如何安装](#如何安装)部分都配置完成后，只需要在项目根目录下分别执行

```bash
python -u PowerPredictServer.py		# 能耗预测服务
python -u PowerEvaluateServer.py	# 能耗评估服务
python -u MetricEvaluateServer.py	# 指标评估服务
```

这三个服务的端口号分别为 50051、50052、50053。你也可以在对应的文件中进行修改。

## 参数详解

这里将对三个proto中的参数进行解释，方便之后向服务发起请求。

能耗预测 [`proto/powerpredict.proto`](./proto/powerpredict.proto)

- host: 预测实体 id
- start: 预测所需**历史数据**起始时间戳
- end: 预测所需**历史数据**结束时间戳
- algorithm: 本次预测所采用的算法，有
  - RF / rf: 随机森林
  - AIMIA / arima
  - LSTMx2 / lstmx2: 堆叠式LSTM
  - TCN / tcn: 时序卷积网络
  - DARNN / darnn: 基于两阶段注意力机制的循环神经网络
- type: 任务类型，有以下两个选项
  - server: 物理机能耗预测任务
  - pod: pod能耗预测任务

能耗评估 [`proto/powerevaluate.proto`](./proto/powerevaluate.proto)

- host: 预测实体 id
- hostType: 任务类型，有以下两个选项server: 物理机能耗预测任务pod: pod能耗预测任务
- start: 评估的起始时间戳
- end: 评估的结束时间戳
- algorithm: 本次预测所采用的算法，有
  - RF / rf: 随机森林
  - AIMIA / arima
  - LSTMx2 / lstmx2: 堆叠式LSTM
  - TCN / tcn: 时序卷积网络
  - DARNN / darnn: 基于两阶段注意力机制的循环神经网络

指标评估 [`proto/metricevaluate.proto`](./proto/metricevaluate.proto)

- type: 任务类型，有以下两种类型
  - qos: 物理机 QoS 指标评估
  - dc: 数据中心能效等级评估
- host: 任务对应的实体
  - type 为 qos: 物理机 id
  - type 为 dc: 数据中心 id（可选）
- start: 评估的起始时间戳
- end: 评估的结束时间戳
- algorithm: 任务采用的算法
  - type 为 qos: brb
  - type 为 dc: membership

## 功能模块

###能耗预测



### 能耗评估



### 指标评估



