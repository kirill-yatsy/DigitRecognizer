import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import utils as t_utils
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_data = pd.read_csv('./data/train.csv',dtype = np.float32)
test_data = pd.read_csv('./data/test.csv',dtype = np.float32 )

train_data['label'] = train_data['label'].astype(int)
# test_data['label'] = test_data['label'].astype(int)

# convert to numpy
targets_numpy = train_data.label.values
features_numpy = train_data.loc[:,train_data.columns != "label"].values

# convert to tensor
targets_tensor = torch.from_numpy(targets_numpy).to(device).long()
features_tensor = torch.from_numpy(features_numpy).to(device).long()

# create dataset
dataset = TensorDataset(features_tensor, targets_tensor)
 
# split dataset
rows = train_data.shape[0]
test_size = int(rows*0.2)
train_size = rows - test_size
trainDataset, testDataset = random_split(dataset, [train_size, test_size])

# init loaders
batch_size = 100
trainLoader = DataLoader(trainDataset, batch_size=batch_size)
testLoader = DataLoader(testDataset, batch_size=batch_size)



train_features, train_labels = next(iter(trainLoader))
img = torch.reshape(train_features[0], (28, 28)) 
label = train_labels[0]
plt.imshow(img)
plt.title(str(label))
plt.show()
print(f"Label: {label}")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Linear(32 * 4 * 4, 10)
        # )
         # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 

    def forward(self, x):
        # print('forward1')
        # print(x.size())

        # x = self.flatten(x)
        # print('forward2')
        # logits = self.linear_relu_stack(x)
        # return logits
        print(x.size())
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        
        # flatten
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        
        return out
    
model = NeuralNetwork().to(device)
print(model)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for (X, y) in trainLoader:
        # train_features, train_labels = next(iter(trainLoader))
        X, y = X.to(device), y.to(device)
        print(X.size())
         
        xx = torch.reshape(X, (100, 1, 28, 28)) 
        # Compute prediction error
        pred = model(xx)
        print(123123)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if X[0] % 100 == 0:
            loss, current = loss.item(), (X[0] + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trainDataset, model, loss_fn, optimizer)
    # test(test_dataloader, model, loss_fn)
print("Done!")