# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from scipy import stats
from torch import autograd
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data as Data

NUM_LAYERS = 4
HIDDEN_SIZE = 32
BATCH_SIZE = 32
INPUT_SIZE = 7
TIME_STEP = 10      # rnn time step

def calMse(pred_Y,data_Y):
    diff = pred_Y - data_Y
    num = np.shape(pred_Y)[0]
    mse = np.sum(diff * diff) / num
    return mse

def train_model(X_train,Y_train,X_test,Y_test,epochs):
    min = 100000000
    model = RNN()
    loss_fun = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    data_loader = Data.DataLoader(dataset=Data.TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)
    for step in range(epochs):
        for x, y in data_loader:
            x_train = Variable(x)
            y_train = Variable(y)
            print('Epoch:', step)
            print(type(model))
            prediction = model(x_train.float())
            # prediction = model(X_train)
            print(prediction.shape)
            print(y_train.shape)
            print("pre")
            loss = loss_fun(prediction, y_train.float())
            print("loss")
            optimizer.zero_grad()
            #loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            print('Loss is {}'.format(loss))
            torch.save(model, 'model/lstmx2.pkl')
            for x_test, y_test in test_loader:
                x_test = Variable(x_test)
                pre_test = model(x_test.float())
                print(pre_test.data.numpy()[:,TIME_STEP-1,:].shape)
                print(y_test.data.shape)
                mse = calMse(pre_test.data.numpy()[:,TIME_STEP-1,:].flatten(), y_test.data.numpy().flatten())
                if mse < min:
                    min = mse
                    torch.save(model, 'model/lstmx2.pkl')
                    model_minmse = torch.load('model/lstmx2.pkl')
                    pre_test = model_minmse(x_test.float())
                    print(calMse(pre_test.data.numpy()[:,TIME_STEP-1,:].flatten(), y_test.data.numpy().flatten()))
                    # torch.save(model_minmse, 'model3.pkl')
                break
        if step % 10 == 0:
            print('MSE is {}'.format(mse))
            print('Min is {}'.format(min))

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += -(-window_size // 2)

def extract_segments(data, window_size = 9):
    # 时序数据集data横向拼接，拼接时长为windoe_size大小的时长，segments保存拼接后的特征，labels保存拼接后的能耗
    segments = []
#     segments = np.empty(((window_size + 1),7))
    labels = np.empty((0))
#     print(segments.shape)
#     print(labels.shape)
    for (start,end) in windows(data,window_size):
        if(len(data.loc[start:end]) == (window_size + 1)):
            signal = data.loc[start:end,["cpu.active.percent", "memory.used.percent", "disk.sda.disk_octets.read",
                    "disk.sda.disk_octets.write", "interface.eth0.if_octets.rx", "interface.eth0.if_octets.tx", "pdu.power"]]
            signal = signal.values
#             print(signal.shape)
            segments.append(signal)
#             print(segments.shape)
            labels = np.append(labels,stats.mode(data.loc[start:end, 'userId'])[0][0])
    return segments, labels


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.LSTM(INPUT_SIZE, HIDDEN_SIZE,num_layers=NUM_LAYERS,batch_first=True,dropout=0.1)
        self.rnn2 = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True,dropout=0.1)
        #self.rnn3 = nn.LSTM(32, 32, num_layers=3, batch_first=True, dropout=0.1)
        self.out = nn.Linear(HIDDEN_SIZE, 1)
    def init_hidden(self):
        return (autograd.variable((torch.rand(NUM_LAYERS,BATCH_SIZE,HIDDEN_SIZE))),
                autograd.variable(torch.rand(NUM_LAYERS,BATCH_SIZE,HIDDEN_SIZE)))
    def forward(self, x):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        h_state=self.init_hidden()
        #h_state = None
        r_out1, h_state1 = self.rnn(x, h_state)
        r_out, h_state = self.rnn2(r_out1,h_state1)
        #r_out, h_state = self.rnn3(r_out2,h_state2)
        #r_out = F.dropout(r_out2, p=0.2, training=self.training)
        #print(r_out.size(1))
        outs = []  # save all predictions
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        m = torch.stack(outs, dim=1)
        return torch.stack(outs, dim=1)

def pred_batch_fill(X):
    if X.shape[0] < BATCH_SIZE:
        fill_rows = BATCH_SIZE - X.shape[0]
        fill_np = np.zeros((fill_rows, TIME_STEP * INPUT_SIZE))
        X_new = np.concatenate((fill_np, X))
        return X_new
    return X

def train(segments, labels):
    feature_list = ["cpu.active.percent", "memory.used.percent", "disk.sda.disk_octets.read",
                    "disk.sda.disk_octets.write", "interface.eth0.if_octets.rx", "interface.eth0.if_octets.tx"]
    percent_feature_name = ["cpu.active.percent", "memory.used.percent"]
    stastic_feature_name = ["disk.sda.disk_octets.read","disk.sda.disk_octets.write", "interface.eth0.if_octets.rx", "interface.eth0.if_octets.tx"]
    #percent_feature_name是百分比特征名，stastic_feature_name是统计型特征名
    reshaped_segments = segments.reshape([len(segments), TIME_STEP, INPUT_SIZE])
    print(labels.shape)
    print(reshaped_segments.shape)
    train_x, test_x, train_y, test_y = train_test_split(reshaped_segments, labels, test_size=0.2, random_state=2020)
    # 模型训练
    # construct data loader
    train_model(torch.Tensor(train_x), torch.Tensor(train_y), torch.Tensor(test_x), torch.Tensor(test_y), epochs=20000)

# @profile
def predict(X, model=None, model_path="model/lstmx2.pkl"):
    if model == None:
        model = torch.load(model_path)
    model.eval()
    X = pred_batch_fill(X)
    X = X.reshape([len(X), TIME_STEP, INPUT_SIZE])
    X_var = Variable(torch.Tensor(X))
    # forecast
    out = model(X_var)
    print(out.shape)
    return out.detach().numpy()[-1, -1, 0]

if __name__ == '__main__':
    # X = np.random.random((800, 70))
    # y = np.random.random((800,))
    # X = X.reshape([len(X), TIME_STEP, INPUT_SIZE])
    # train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=2020)
    # print(train_x.shape)
    # train_model(torch.Tensor(train_x), torch.Tensor(train_y), torch.Tensor(test_x), torch.Tensor(test_y), epochs=20000)
    X = np.random.random((1, 70))

    print(X.shape)
    print(pred(torch.Tensor(X)))




    # feature_list = ["cpu.active.percent", "memory.used.percent", "disk.sda.disk_octets.read",
    #                 "disk.sda.disk_octets.write", "interface.eth0.if_octets.rx", "interface.eth0.if_octets.tx"]
    # percent_feature_name = ["cpu.active.percent", "memory.used.percent"]
    # stastic_feature_name = ["disk.sda.disk_octets.read","disk.sda.disk_octets.write", "interface.eth0.if_octets.rx", "interface.eth0.if_octets.tx"]
    # #percent_feature_name是百分比特征名，stastic_feature_name是统计型特征名
    #
    # df_path = ''
    # dataset = pd.read_csv(df_path)
    # dataset = fillna_decompose(dataset)
    # dataset = correct_percent(dataset, percent_feature_name)
    # dataset = correct_statistics(dataset, stastic_feature_name)
    # #读入数据并补全，然后根据数据类型（百分比和统计型数据进行预处理）
    #
    # win_size = 9
    # segments, labels = extract_segments(dataset, win_size)
    # segments = np.array(segments)
    # labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
    # reshaped_segments = segments.reshape([len(segments), 10, 7])
    # print(labels.shape)
    # print(reshaped_segments.shape)
    # train_test_split = np.random.rand(len(reshaped_segments)) < 0.80
    # train_x = reshaped_segments[train_test_split]
    # train_y = labels[train_test_split]
    # test_x = reshaped_segments[~train_test_split]
    # test_y = labels[~train_test_split]
    # model = RNN()
    # loss_fun = torch.nn.MSELoss()
    # # 模型训练
    # train_model(torch.Tensor(train_x), torch.Tensor(train_y), torch.Tensor(test_x), torch.Tensor(test_y), epochs=20000)
    # optimizer = torch.optim.Adam(model.parameters())
