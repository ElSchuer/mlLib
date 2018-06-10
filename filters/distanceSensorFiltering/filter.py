import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

def generateNoisyMeasurementsWithPeaks(x0, dx, count, noiseFactor, peakStep, peakGain):
    data = generateNoisyMeasurements(x0, dx, count, noiseFactor)
    return [data[i]+peakGain*random.choice([1, -1]) if i % peakStep == 0 else data[i] for i in range(len(data))]

def generateNoisyMeasurements(x0, dx, count, noiseFactor):
    return [x0 + dx*t + np.random.randn()*noiseFactor for t in range(count)]

def plotTimeData(x, t):
    plt.plot(t, x)
    plt.xlabel('time (s)')
    plt.ylabel('distance (cm)')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    data = generateNoisyMeasurementsWithPeaks(x0=30, dx=3, count=100, noiseFactor=10, peakStep=25, peakGain=100)
    plotTimeData(data, np.arange(0, 100, 1))