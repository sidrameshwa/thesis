import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import namedtuple
from torchvision.models import resnet18

class Data(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        
        x = self.x[i]
        y = self.y[i]

        return x, y

# Deep Regression model 
class RegressionModel(nn.Module):
    def __init__(self, input_size = 10):
        super(RegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

# LSTM model
class MyLSTM(nn.Module):
    def __init__(self, hidden_size=64, input_size = 10):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x.shape = (batch_size, sequence_length, input_size)
        output, (hidden, cell) = self.lstm(x)
        # output.shape = (batch_size, sequence_length, hidden_size)
        # hidden.shape = (1, batch_size, hidden_size)
        # cell.shape = (1, batch_size, hidden_size)
        output = self.linear(hidden.squeeze().double())
        # output.shape = (batch_size, 1)
        return output

#RNN model   
class RNN(nn.Module):
    def __init__(self, input_size = 10, hidden_size=64, output_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).double()
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=1, hidden_dim=32, num_layers=2, num_heads=2, dropout=0.3):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
        x = self.fc(x[:, -1, :])
        return x


class ResLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, dropout=0.3):
        super(ResLSTM, self).__init__()

        # ResNet block as feature extractor
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # LSTM to process the extracted features
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer for prediction
        self.fc = nn.Linear(hidden_size, 1)
        self.compressor = nn.Linear(512, 10)

    def forward(self, x):
        # Reshape input data to match ResNet input
        x = x.unsqueeze(1)
        x = self.resnet(x)
        # Pass ResNet features through LSTM
        x = self.compressor(x)
        x = x.view(x.size(0), 1, -1)
        x, _ = self.lstm(x)

        # Get the last output from LSTM and pass it through a fully connected layer for prediction
        x = self.fc(x[:, -1, :])

        return x