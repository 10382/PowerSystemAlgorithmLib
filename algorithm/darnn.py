import torch
import numpy as np
from torch import nn
from torch.nn import functional as tf
from torch import optim
from torch.utils import data as Data

ENCODER_HIDDEN_SIZE = 64
DECODER_HIDDEN_SIZE = 64
SEQ_LEN = 10
INPUT_SIZE = 7
OUT_FEAS = 1
LR = 0.01
EPOCHS = 1000
BATCH_SIZE = 128
MODEL_PATH = "model/darnn.pkl"
# MODEL_PATH = "../model/darnn.pkl"

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

def init_hidden(x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return torch.zeros(1, x.size(0), hidden_size)


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int, device):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.device = device
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T, out_features=1)

    # @profile
    def forward(self, input_data):
        # input_data: (batch_size, T, input_size)
        # print(type(input_data))
        input_weighted = torch.zeros(input_data.size(0), self.T, self.input_size, device=self.device)
        input_encoded = torch.zeros(input_data.size(0), self.T, self.hidden_size, device=self.device)
        # hidden, cell: initial states with dimension hidden_size
        hidden = init_hidden(input_data, self.hidden_size).to(self.device)  # 1 * batch_size * hidden_size
        cell = init_hidden(input_data, self.hidden_size).to(self.device)

        print(input_data.dtype)
        for t in range(self.T):
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2).to(self.device)  # batch_size * input_size * (2*hidden_size + T - 1)
            # Eqn. 8: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T))  # (batch_size * input_size) * 1
            # Eqn. 9: Softmax the attention weights
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)  # (batch_size, input_size)
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size, decoder_hidden_size, T, out_feats, device):
        super(Decoder, self).__init__()
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()
        self.device = device

    # @profile
    def forward(self, input_encoded, y_history):
        # input_encoded: (batch_size, T, encoder_hidden_size)
        # y_history: (batch_size, T)
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        hidden = init_hidden(input_encoded, self.decoder_hidden_size).to(self.device)
        cell = init_hidden(input_encoded, self.decoder_hidden_size).to(self.device)
        context = torch.zeros(input_encoded.size(0), self.encoder_hidden_size, device=self.device)

        for t in range(self.T):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2).to(self.device)
            # Eqn. 12 & 13: softmax on the computed attention weights
            x = tf.softmax(
                    self.attn_layer(
                        x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                    ).view(-1, self.T),
                    dim=1)  # (batch_size, T)

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)

            # Eqn. 15
            # print(context.shape, y_history.shape)
            y_tilde = self.fc(torch.cat((context, y_history[:, t:t+1]), dim=1))  # (batch_size, out_size)
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        return self.fc_final(torch.cat((hidden[0], context), dim=1))

class DaRnn(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T, input_size, out_feas, learning_rate, device):
        super(DaRnn, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.T = T
        self.input_size = input_size
        self.out_feas = out_feas
        self.learning_rate = learning_rate
        self.device = device

        self.Encoder = Encoder(self.input_size, self.encoder_hidden_size, self.T, device)
        self.Decoder = Decoder(self.encoder_hidden_size, self.decoder_hidden_size, self.T, self.out_feas, device)

        self.criterion = nn.MSELoss()
        self.encoder_optimizer = optim.Adam(self.Encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(self.Decoder.parameters(), lr=learning_rate)

    # @profile
    def forward(self, X, y_history):
        input_weighted, input_encoded = self.Encoder(X)
        # print(y_history.shape)
        y_pred = self.Decoder(input_encoded, y_history)
        return y_pred

# @profile
def train(X, Y, epochs=1000, lr=0.01, verbose=1):
    net = DaRnn(ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, SEQ_LEN, INPUT_SIZE, OUT_FEAS, LR, DEVICE).to(DEVICE)
    X, Y_history = data_preprocess(X, SEQ_LEN, INPUT_SIZE)
    # print(Y_history.shape)
    dataset = MyDataset(X, Y_history, Y)
    train_loader = Data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=True)
    criterion = torch.nn.MSELoss()
    loss_min = 10000000
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            x, y_history, y = data
            # print(y_history.shape)
            x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
            y_history = torch.tensor(y_history, dtype=torch.float32).to(DEVICE)
            y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
            net.encoder_optimizer.zero_grad()
            net.decoder_optimizer.zero_grad()
            y_pred = net(x, y_history)
            # loss = torch.tensor(0)
            loss = criterion(y, y_pred)
            loss.backward()
            net.encoder_optimizer.step()
            net.decoder_optimizer.step()
        print(loss, loss.item())
        if loss.item() < loss_min:
            torch.save(net, MODEL_PATH)
            loss_min = loss.item()
        if verbose:
            if epoch % 20 == 0:
                print('epoch: ', epoch, 'loss', loss.item())

def pred_batch_fill(X):
    if X.shape[0] < BATCH_SIZE:
        fill_rows = BATCH_SIZE - X.shape[0]
        fill_np = np.zeros((fill_rows, SEQ_LEN * INPUT_SIZE))
        X_new = np.concatenate((fill_np, X))
        return X_new
    return X

# @profile
def predict(x, model=None):
    x = pred_batch_fill(x)
    x, y_history = data_preprocess(x, SEQ_LEN, INPUT_SIZE)
    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    y_history = torch.tensor(y_history, dtype=torch.float32).to(DEVICE)
    # print(x.shape)
    # print(y_history.shape)
    # X_padding = torch.zeros(BATCH_SIZE, x.shape[1], x.shape[2], dtype=float).to(DEVICE)
    # Y_history_padding = torch.zeros(BATCH_SIZE, y_history.shape[1], dtype=float).to(DEVICE)
    # print(Y_history_padding.shape)
    # X_padding[-1] = x
    # Y_history_padding[-1] = y_history
    if model == None:
        model = torch.load(MODEL_PATH)
    pred = model(x, y_history).detach().cpu().numpy()[-1]
    return pred[0]


def gt_history(data, his_len):
    y_history = list()
    for idx in range(len(data)):
        end_idx = idx + his_len
        if end_idx > len(data)-1:
            break
        history = data[idx: end_idx]
        y_history.append(history)
    return np.array(y_history)


class MyDataset(Data.Dataset):
    def __init__(self, X, Y_history, Y_target):
        super(MyDataset, self).__init__()
        self.X = X
        self.Y_target = Y_target
        self.Y_history = Y_history

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        data = (self.X[idx], self.Y_history[idx], self.Y_target[idx])
        return data


def data_preprocess(X, seq_len, feas):
    X = X.reshape(-1, seq_len, feas)
    # Y_history = gt_history(X, seq_len)
    Y_history = X[:, :, -1]
    # print(Y_history.shape)
    return X, Y_history

if __name__ == '__main__':
    X = np.random.random((1, 70))
    print(X.shape)
    print(predict(X))