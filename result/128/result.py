import numpy as np
import matplotlib.pyplot as plt


train = np.load('kernel=3layer=20channel=64trainLossRecord.npy')
SNR1 = 10*np.log10(1/np.sqrt(train))

validation = np.load('kernel=3layer=20channel=64validationLossRecord.npy')
SNR2 = 10*np.log10(1/np.sqrt(validation))
loss = min(train)
print(10*np.log10(1/(1/128)))
print(10*np.log10(1/np.sqrt(loss)))
print(np.argmin(train))
print(10*np.log10((1/128)/np.sqrt(loss)))
plt.figure()
plt.plot(train, label='train loss')
plt.plot(validation, label='validation loss')
# plt.ylim(0.002, 0.004)
plt.legend()
plt.figure()
plt.plot(SNR1, label='train')
plt.plot(SNR2, label='validation')
# plt.ylim(10, 14)
plt.show()