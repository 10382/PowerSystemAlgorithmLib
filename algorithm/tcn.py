import torch
from torch import nn, optim
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from torch.utils import data as Data


# 一些常量的定义，开始
HISTORY = 10
BATCH_SIZE = 128
NUM_CHANNELS = [32, 16, 8, 4, 2, 1]
KERNEL_SIZE = 3
DROPOUT = 0.2
LR = 0.01
EPOCHS = 300
# 一些常量的定义，结束

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

# 模型部分，开始
class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # 由于补零可能会导致time_step变化，因此设计chomp1d类保证time_step不变
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # Residual Block
        self.conv1 = weight_norm(nn.Conv1d(in_channels=n_inputs, out_channels=n_outputs, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(in_channels=n_outputs, out_channels=n_outputs, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(in_channels=n_inputs, out_channels=n_outputs, kernel_size=1) if n_inputs != n_outputs else None

        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(n_inputs=in_channels, n_outputs=out_channels, kernel_size=kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1)*dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
# 模型部分，结束

# @profile
# 辅助方法，开始
def train_for_tcn(model, criterion, optimizer, train_loader, epochs):
    model.train()
    loss_min = 100000000
    for epoch in range(1, epochs + 1):
        batch_count = 0
        nrmse = 0.0
        for x, y in train_loader:
            var_x = Variable(x)
            var_y = Variable(y)
            # forward
            out = model(var_x)
            prediction = torch.mean(out, dim=2)
            loss = torch.sqrt(criterion(prediction, var_y)) / torch.std(var_y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # information
            batch_count += 1
            nrmse += loss.item()
        print('epoch: %d,\tnrmse: %.4f' % (epoch, nrmse / batch_count))
        # if loss.item() < loss_min:
        #     loss_min = loss.item()
        #     torch.save(model, 'model/tcn.pkl')
    return model

# 确定湘潭的交付时间，什么时候问了返回了什么结果

# @profile
def test_for_tcn(model, x):
    model.eval()
    var_x = Variable(x)
    out = model(var_x)
    prediction = torch.mean(out, dim=2)
    return prediction
# 辅助方法，结束


# 对外提供的方法，开始
def train(X, y):
    # construct data loader
    X_tensor = torch.Tensor(X).reshape(X.shape[0], -1, HISTORY).to(DEVICE)
    num_inputs = X_tensor.size(1)
    y_tensor = torch.Tensor(y).reshape(-1, 1).to(DEVICE)
    data_loader = Data.DataLoader(dataset=Data.TensorDataset(X_tensor, y_tensor), batch_size=BATCH_SIZE, shuffle=False)

    # define model, criterion, optimizer
    model = TemporalConvNet(
        num_inputs=num_inputs,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT
    ).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=LR)

    # train and save model
    model = train_for_tcn(model=model, criterion=criterion, optimizer=optimizer, train_loader=data_loader, epochs=EPOCHS)
    return True

# @profile
def predict(X, model=None):
    # construct tensor
    X_tensor = torch.Tensor(X).reshape(X.shape[0], -1, HISTORY).to(DEVICE)

    # load model
    if model == None:
        model = torch.load('model/tcn.pkl')

    # forecast
    result = test_for_tcn(model=model, x=X_tensor)
    # print(result)
    return result.cpu().detach().numpy()[0][0]
# 对外提供的方法，结束

if __name__ == '__main__':
    import numpy as np

    X = np.random.random((800, 70))
    y = np.random.random((800,))

    train(X, y)
    prediction = predict(X)
    print(prediction)