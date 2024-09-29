import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import os

torch.manual_seed(5)
minEpoch = 200
maxEpoch = 1000
maxWait = 5
batchSize = 128
lr = 3e-4
saveModel = False
kernelSize = 3
channel = 64
layers = 20
dataType = '\\128\\'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
criterion = nn.MSELoss().to(device)

# data load
dataPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "\\data" + dataType
trainInput = torch.unsqueeze(torch.from_numpy(np.load(dataPath+"trainInput.npy")).float(), 1)
trainLabel = torch.unsqueeze(torch.from_numpy(np.load(dataPath+"trainLabel.npy")).float(), 1)
trainData = TensorDataset(trainInput, trainLabel)
trainData = DataLoader(trainData, batch_size=batchSize, shuffle=True, pin_memory=True)
validationInput = torch.unsqueeze(torch.from_numpy(np.load(dataPath+"validationInput.npy")).float(), 1)
validationLabel = torch.unsqueeze(torch.from_numpy(np.load(dataPath+"validationLabel.npy")).float(), 1)
validationData = TensorDataset(validationInput, validationLabel)
validationData = DataLoader(validationData, batch_size=1, shuffle=False, pin_memory=True)


# net
class DnCNN(nn.Module):
    def __init__(self):
        super().__init__()
        padding = (kernelSize - 1) // 2
        self.input = nn.Sequential(
            nn.Conv1d(1, channel, kernelSize, padding=padding, bias=False),
            nn.ReLU(inplace=True))
        middle = []
        for _ in range(layers - 2):
            middle.append(nn.Conv1d(channel, channel, kernelSize, padding=padding, bias=False))
            middle.append(nn.BatchNorm1d(channel))
            middle.append(nn.ReLU(inplace=True))
        self.middle = nn.Sequential(*middle)
        self.out = nn.Conv1d(channel, 1, kernelSize, padding=padding, bias=False)

    def forward(self, x):
        return x - self.out(self.middle(self.input(x)))


# training process
net = DnCNN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
trainLossRecord = []
validationLossRecord = []
netBest = copy.deepcopy(net)
bestEpoch = 0
bestLoss = 1.0
startTime = time.time()
wait = 0
for i in range(maxEpoch):
    net.train()
    trainLoss = 0.0
    for j, data in enumerate(trainData):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        trainLoss += loss.item()
    net.eval()
    validationLoss = 0.0
    for j, data in enumerate(validationData):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        validationLoss += loss.item()

    trainLoss = trainLoss/(len(trainData))
    validationLoss = validationLoss/(len(validationData))
    endTime = time.time()
    print(f'Epoch{i + 1} Training time:{int(endTime - startTime)}s trainPSNR:{10 * np.log10(1 / np.sqrt(trainLoss)):.3f}'
          f' valPSNR:{10 * np.log10(1 / np.sqrt(validationLoss)):.3f}')
    trainLossRecord.append(trainLoss)
    validationLossRecord.append(validationLoss)
    # save best net
    if bestLoss > validationLoss:
        bestLoss = validationLoss
        netBest = copy.deepcopy(net)
        bestEpoch = i + 1
        wait = 0
    else:
        wait += 1
    if saveModel:
        savePath = '.' + '\\result' + dataType + 'kernel=' + str(kernelSize) + 'layer=' + str(layers) + 'channel=' + str(channel)
        torch.save(netBest, savePath + '.pkl')
        np.save(savePath + 'trainLossRecord.npy', trainLossRecord)
        np.save(savePath + 'validationLossRecord.npy', validationLossRecord)
    # early stop
    if wait >= maxWait and i > minEpoch:
        break

endTime = time.time()
print(f"Training time:{int(endTime - startTime)}s")
print(f'bestEpoch:{bestEpoch},'
      f'trainPSNR:{10*np.log10(1/np.sqrt(trainLossRecord[bestEpoch-1])):.3f}'
      f' valPSNR:{10*np.log10(1/np.sqrt(bestLoss)):.3f}')
