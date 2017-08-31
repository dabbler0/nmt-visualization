import math
import json
import numpy
import sys
import scipy.cluster as cluster
import scipy.spatial as spatial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.serialization import load_lua

# Need to increase recursion limit for hierarchical clustering
sys.setrecursionlimit(15000)

if len(sys.argv) < 3:
    print('Need arguments: description list and output file')

# Memoization function, for speed
def _memoize(f):
    d = {}
    def memoized(*args):
        args = tuple(args)
        print(args)
        if args not in d:
            d[args] = f(*args)

        return d[args]
    return memoized

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Load all descriptions from the description file,
# which should be cmdline argument [1]
print('Reading data')
sys.stdout.flush()
files = open(sys.argv[1]).read().split('\n')[:-1]

# Load out all the description files
data = {}
for filename in files:
    print('Loading %s' % (filename,))
    data[filename] = [x.numpy().tolist() for x in load_lua(filename)['decodings']]

keys = sorted(data.keys())
print(keys)

# For each neuron, get a big array describing activations
# over all known tokens (rather than an array of arrays per line)
print('Getting neurons')
sys.stdout.flush()
all_arrays = []
for key in keys:
    data[key] = numpy.concatenate([numpy.array(line) for line in data[key]])
    data[key] = data[key][:, :500]
    all_arrays.append(data[key])

# Stack all the neurons together.
# We should now have a (500 * num_networks) x (sample_size) matrix.
all_neurons = numpy.concatenate(all_arrays, 1)

# Get labels for all the neurons in xx-yy-z:n form
names = []
for key in keys:
    for i in range(500):
        names.append('%s:%d' % (key.split('/')[-1].split('.')[0], i))

# Get correlations
print('Getting correlations')
distances = spatial.distance.pdist(
    numpy.transpose(all_neurons),
    metric = 'correlation'
)

# Take absolute values of correlation
distances = 1 - numpy.absolute(1 - distances)

# Ask sklearn to do correlations
linkages = cluster.hierarchy.linkage(
    distances,
    metric = 'correlation'
)

# Create a dendrogram
plt.figure(figsize = (10, 500))
axes = plt.gca()
dendrogram = cluster.hierarchy.dendrogram(linkages, ax=axes, labels=names, distance_sort=True, orientation='right')
plt.savefig(sys.argv[2], format='svg')
