import torch
from torch.utils.serialization import load_lua
import numpy

# Load descriptions
print('Loading descriptions...')
descriptions = load_lua(sys.argv[1])

# Go through all neurons
all_neurons = []
for neuron in range(500):

    # For each neuron, find the mean square distance from
    # the activation at each position to the mean for that position.

    # This is the MSE of predicting the activations from the positions.


    # First, find the mean at each position, as well as the mean
    # of the neuron
    neuron_mean = 0
    neuron_total = 0
    mapping = {}
    for line in descriptions:
        for i, token in enumerate(line.numpy()):
            if i not in mapping:
                mapping[i] = []
            mapping[i].append(token[neuron])
            neuron_mean += token[neuron]
            neuron_total += 1
    neuron_mean /= neuron_total

    # Then, find the mean square distance to each position, as well as the
    # standard deviation of the neuron
    neuron_stdev = 0
    error = 0
    for x in mapping:
        mapping[x] = numpy.array(mapping[x])
        error += ((mapping[x] - mapping[x].mean()) ** 2).sum()
        neuron_stdev += ((mapping[x] - neuron_mean) ** 2).sum()

    # Divide by the stdev to get what the error would be for the
    # normalized neuron
    error /= neuron_stdev

    all_neurons.append((error, neuron))

# Sort neurons by predictor accuracy
all_neurons = sorted(all_neurons)

# Print.
print(all_neurons)
