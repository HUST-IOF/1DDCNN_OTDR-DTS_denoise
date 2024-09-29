"""

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 超参数
layers = 20
torch.manual_seed(5)
kernelSize = 3
channel = 64

sampleInter = 1  # sampling interval 1m
length = 10000  # fiber length 10km
samplePoints = round(length / sampleInter)
totalPoints = samplePoints + 1000  # Points without fiber optic scattering
otdrStart = 1
alpha = -0.3  # fiber loss 0.3dB/km
noise = 1/128  # noise std


orOTDR = np.zeros(totalPoints)
for i in range(samplePoints):
    orOTDR[i] = otdrStart * 10 ** (alpha * (i * sampleInter / 1000) / 10)

orOTDR[5000:] = orOTDR[5000:] * 10 ** (-3 / 10)
orOTDR[6000] = orOTDR[6000]*2
# orOTDR[6002] = orOTDR[6002]*2


Input = orOTDR + np.random.randn(totalPoints) * noise
Label = orOTDR
Input = torch.unsqueeze(torch.from_numpy(Input).float(), 0)
Input = torch.unsqueeze(Input, 1)


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


net = torch.load('kernel=' + str(kernelSize) + 'layer=' + str(layers) + 'channel=' + str(channel) + '.pkl', map_location=torch.device('cpu'))
net.eval()
Output = net(Input)
plt.figure()
plt.plot(np.arange(totalPoints)*sampleInter, Input.detach().numpy()[0, 0, ], '-', label="input")
plt.plot(np.arange(totalPoints)*sampleInter, Output.detach().numpy()[0, 0, ], '-', label="output")
plt.plot(np.arange(totalPoints)*sampleInter, Label, '-', label="label")
plt.legend(loc=1)
plt.xlabel('Fiber Length (m)')
plt.ylabel('Intensity (a.u.)')
print(f'inPSNR:{10 * np.log10(1 / np.std(Input.detach().numpy()[0, 0, ]-Label)):.3f}'
      f' outPSNR:{10 * np.log10(1 / np.std(Output.detach().numpy()[0, 0, ]-Label)):.3f}')

plt.show()


