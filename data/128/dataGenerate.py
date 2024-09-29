"""
random intensity duration 1~50 points
noise std = 1/128
pointEachTrace = 10k
traceNumber = 10k
"""
import numpy as np
import matplotlib.pyplot as plt
import time

# noise
std = 1/128
# trace generate
np.random.seed(5)
pointEachTrace = 10000
traceNumber = 10000
trainPercentage = 0.8
trainTrace = round(traceNumber * trainPercentage)
validationTrace = round(traceNumber * (1 - trainPercentage))
trainInput = np.zeros([trainTrace, pointEachTrace])
trainLabel = np.zeros([trainTrace, pointEachTrace])
validationInput = np.zeros([validationTrace, pointEachTrace])
validationLabel = np.zeros([validationTrace, pointEachTrace])
startTime = time.time()
for n in range(traceNumber):
    i = 0
    original = np.zeros(pointEachTrace)
    while i < pointEachTrace:
        j = np.random.randint(1, 51)
        if i + j < pointEachTrace:
            original[i:i+j] = np.ones(j)*np.random.rand()
            i += j
        else:
            original[i:] = np.ones(pointEachTrace - i) * np.random.rand()
            break
    # add noise
    noisy = original + np.random.randn(pointEachTrace)*std

    if n+1 <= trainTrace:
        trainInput[n, ] = noisy
        trainLabel[n, ] = original
    else:
        validationInput[n-trainTrace, ] = noisy
        validationLabel[n-trainTrace, ] = original

    if (n+1) % (traceNumber/100) == 0:
        timeNow = time.time()
        timeRes = int((timeNow-startTime)*(traceNumber/(n+1)-1)/60)
        print("\r", f'Generate data progress {int((n+1)/traceNumber*100)}% remaining time：{timeRes}minute', end="", flush=True)

np.save('trainInput.npy', trainInput)
np.save('trainLabel.npy', trainLabel)
np.save('validationInput.npy', validationInput)
np.save('validationLabel.npy', validationLabel)


# plt.plot(trainLabel[0], label="original")
# plt.plot(trainInput[0], label="noisy")
# plt.plot(trainInput[0]-trainLabel[0], label="noisy")
# plt.legend(loc=0)
plt.show()
