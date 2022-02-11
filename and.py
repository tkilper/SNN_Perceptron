import numpy as np
import lif
import matplotlib.pyplot as plt


# 2 input neurons
# 1 output neuron
# initialize random weights
# Adjust weights using hebbian


# https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007692#pcbi-1007692-g002
# At the end of learning, the neuron’s tuning curves are uniformally distributed (Fig 2Giii), and the quality of the representation becomes optimal for all input signals (Fig 2Aiii and 2Ciii).
# What are decoding weights?
# What are tuning curves?
# Are our inputs correlated? (for AND, OR gate)
# When does learning converge? Mainly what does this mean: "Learning converges when all tuning curve maxima are aligned with the respective feedforward weights (Fig 3Bii; dashed lines and arrows)."


# ---------------------------------------------------------------------------


# https://www.geeksforgeeks.org/single-neuron-neural-network-python/
class SNN():

    def __init__(self):
        np.random.seed(1)  # Generate same random weights for every trial
        # Matrix containing the weights between each neuron
        self.z = np.array[1,0,0,0,1,1,0,1,0,1]
        self.alpha = np.array[1,1,1]
        self.beta = np.array[2,2,2]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, inputs):

        out = np.dot(inputs, self.weights)

        zhat = SNN.sigmoid(out)

        error = self.z - zhat
        return zhat

    # pass spikes from input to output
    # pass the out of the input to the in of the output (the weights affect this process)
    # how do the weights effect this connection?

    def train(self, inputs, outputs, epochs):
        weights = np.random.rand(2, 1)

        # perform encoding into input neurons (input_nns)

        for i in range(epochs):
            output_nns = self.forward_propagation(inputs)

            delta_w1 = np.dot([inputs[0]*output_nns, input[0], output_nns], self.alpha)  # dot product?  multiply the avg rates?
            delta_w2 = np.dot([inputs[1]*output_nns, input[1], output_nns], self.beta)

            weights[0] += delta_w1
            weights[1] += delta_w2

        # loop through each combo of x and y     ?
        # use hebbian rule to determine weight adjustment
        # apply weight adjustment


# https://praneethnamburi.com/2015/02/05/simulating-neural-spike-trains/
# fr:  firing rate estimate  (in Hz)
# train_length:  length of the spike train (in seconds)
def poissonSpike(fr, nbins, num_trials):
    dt = 1 / 1000
    spikeMatrix = np.random.rand(num_trials, nbins) < fr * dt
    t = np.arange(0, (nbins * (dt - 1)), dt)
    return (spikeMatrix, t)


def rasterPlot(spikeMatrix):
    spikes_x = []
    spikes_y = []
    for i in range(spikeMatrix.shape[0]):
        for j in range(spikeMatrix.shape[1]):
            if spikeMatrix[i][j]:
                spikes_y.append(i)
                spikes_x.append(j)

    plt.scatter(spikes_x, spikes_y, marker="|")
    plt.yticks(np.arange(np.amax(spikes_y)))
    # plt.xticks(np.arange(np.amax(spikes_x)))
    plt.xlabel("Time Step")
    plt.ylabel("Trial")
    plt.show()


sm, t = poissonSpike(300, 30, 20)

rasterPlot(sm)

#