import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device = torch.device('mps') # note: change to mps once I run on computer

data_df = pd.read_csv('/Users/avyuktgupta/Desktop/Projects/Current Projects/rice/riceClassification.csv')
#data_df.head()

data_df.dropna(inplace=True)

data_df.drop(['id'], axis=1, inplace=True)

#print(data_df.shape)

data_df.head()

#print(data_df['Class'].unique())

data_df['Class'].value_counts()

original_df = data_df.copy()

for column in data_df.columns:
  data_df[column] = data_df[column]/data_df[column].abs().max()

#data_df

X = np.array(data_df.iloc[:, :-1])
Y = np.array(data_df.iloc[:, -1])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
"""
X_train.shape

X_test.shape

X_test.shape

y_test.shape
"""
class dataset(Dataset):
  def __init__(self, x, y):
    self.X = torch.tensor(x, dtype=torch.float32).to(device)
    self.Y = torch.tensor(y, dtype=torch.float32).to(device)
  def __len__(self):
    return len(self.X)
  def __getitem__(self, index):
    return self.X[index], self.Y[index]

training_data = dataset(X_train, y_train)
validation_data = dataset(X_val, y_val)
testing_data = dataset(X_test, y_test)

train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=32, shuffle=True)

BATCH_SIZE = 32
EPOCHS = 15
HIDDEN_NEURONS = 10
LR = 1e-3

class model(nn.Module):
  def __init__(self):
    super(model, self).__init__()
    self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
    self.linear = nn.Linear(HIDDEN_NEURONS, 1)
    self.norm = nn.LayerNorm(HIDDEN_NEURONS)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()
  def forward(self, x):
    x = self.input_layer(x)
    x = self.norm(x)
    x = self.relu(x)
    x = self.linear(x)
    x = self.sigmoid(x)
    return x

network = model().to(device)

#summary(network, (X.shape[1],))

criterion = nn.BCELoss()
optimizer = Adam(network.parameters(), lr=LR)

total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []
network.load_state_dict(torch.load('train.pth'))
for epoch in range(EPOCHS):
  total_acc_train = 0
  total_loss_train = 0
  total_acc_val = 0
  total_loss_val = 0

  for data in train_dataloader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    pred = network(inputs).squeeze(1)
    batch_loss = criterion(pred, labels)

    total_loss_train += batch_loss.item()
    acc = ((pred).round() == labels).sum().item()
    total_acc_train += acc

    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  with torch.no_grad():
    for data in validation_dataloader:
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      pred = network(inputs).squeeze(1)
      batch_loss = criterion(pred, labels)

      total_loss_val += batch_loss.item()
      acc = ((pred).round() == labels).sum().item()
      total_acc_val += acc

  total_loss_train_plot.append(round(total_loss_train/1000, 4))
  total_loss_validation_plot.append(round(total_loss_val/1000, 4))
  total_acc_train_plot.append(round(total_acc_train/training_data.__len__() * 100, 4))
  total_acc_validation_plot.append(round(total_acc_val/validation_data.__len__() * 100, 4))

  print(f'Epoch: {epoch+1} Train Loss: {round(total_loss_train/1000, 4)} Train Accuracy: {round(total_acc_train/training_data.__len__() * 100, 4)} Validation Loss: {round(total_loss_val/1000, 4)} Validation Accuracy: {round(total_acc_val/validation_data.__len__() * 100, 4)}')
  print('='*110)

torch.save(network.state_dict(), 'train.pth')

with torch.no_grad():
  total_loss_test = 0
  total_acc_test = 0
  for data in test_dataloader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    pred = network(inputs).squeeze(1)
    batch_loss_test = criterion(pred, labels).item()
    total_loss_test += batch_loss_test
    acc=((pred).round() == labels).sum().item()
    total_acc_test += acc
print(f'Accuracy: {round(total_acc_test/testing_data.__len__() * 100, 4)}')
