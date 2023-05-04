import mlflow
import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import namedtuple
from torchvision.models import resnet18
import livelossplot
from livelossplot import PlotLosses
from sklearn.metrics import mean_squared_error
import argparse
from models import *
import dagshub

# Reading csv file 
data = pd.read_csv('data/Data-set.csv')

#converting to list
data = data.value.values

#80% for training and 20% for testing
# data preparation for training and testing
train_index = int(len(data)*0.8)

train = data[:train_index].reshape(-1,1)

test = data[train_index:].reshape(-1,1)

# standardize data

scaler = preprocessing.StandardScaler().fit(train)

train = scaler.transform(train)

test = scaler.transform(test)

# create data with a sliding window size 10 and step 1
def create_sliding_window_data(arr, window_size=10):
    X, y = [], []
    for i in range(len(arr)-window_size):
        # Create a window of size 10
        window = arr[i:i+window_size]
        # Append the window to the input X
        X.append(window)
        # Append the next element as the output y
        y.append(arr[i+window_size])
    
    # Convert the input and output to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X.squeeze(), y.squeeze()




def trainer(params):
  X_train, y_train = create_sliding_window_data(train, window_size = params['window_size'])
  X_test, y_test = create_sliding_window_data(test, window_size = params['window_size'])

  X_train = torch.from_numpy(X_train).double()
  y_train = torch.from_numpy(y_train).double()

  train_dataset = Data(X_train, y_train)
  train_loader = DataLoader(train_dataset, batch_size = params['batch_size'], shuffle=True)

  test_dataset = Data(X_test, y_test)
  test_loader = DataLoader(test_dataset, batch_size = params['batch_size'], shuffle=True)

  if params["model"] =="resnet":
    model = ResLSTM().double()
  elif params["model"] == "transformer":
    model = TransformerModel().double()
  elif params["model"] == "rnn":
    model = RNN().double()
  elif params["model"] == "lstm":
    model = MyLSTM().double()
  else:
    model = RegressionModel().double()

  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr= params['lr'])

  dagshub.init("thesis", "sidrameshwar", mlflow=True)
  with mlflow.start_run() as run:  
    for key, value in params.items():
        mlflow.log_param(key, value)

    print('Deep Learning models')
    print('====================')

    print('Training phase')

    liveloss = PlotLosses()
    for epoch in range(params['epochs']):

        print("Active Run ID: %s, Epoch: %s \n" % (run.info.run_uuid, epoch+1))
        total_loss = 0
        logs = {}
        i = 0
        for X_train, y_train in train_loader:
          i += 1
          optimizer.zero_grad()

          outputs = model(X_train.view(X_train.shape[0], 1, X_train.shape[-1]))
          #outputs = model(X_train)
          loss = criterion(outputs.squeeze(), y_train)
          total_loss += loss.item()
          loss.backward()
          optimizer.step()
        print('Epoch {}, Loss: {:.4f}'.format(epoch+1, total_loss/i))
        logs['MSE loss'] = total_loss/i 
        mlflow.log_metric('train_loss', total_loss/i, step=epoch)

    torch.save(model.state_dict(), f"{params['model']}_{params['epochs']}_{params['batch_size']}_{params['window_size']}.pth")
    
    print('Test Phase')

    with torch.no_grad():
        total_loss = 0
        i = 0
        for X_test, y_test in test_loader:
          i += 1
          outputs = model(X_test.view(X_test.shape[0], 1, X_test.shape[-1]))
          #outputs = model(X_test)
          
          total_loss += criterion(outputs, y_test)
        print('Loss: {:.4f}'.format(total_loss.item()/i))
        mlflow.log_metric('test_loss', total_loss/i, step=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001 , help="learning rate")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--model", type=str, default='transformer', help='different models')

    args = parser.parse_args()
    params = {'lr' : args.lr, 'batch_size': args.batch_size, 'epochs':args.epochs, 'window_size':args.window_size, 'model': args.model}
    trainer(params)
