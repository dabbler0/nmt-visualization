from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from urlparse import urlparse
import json

import colorsys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.serialization import load_lua
import numpy

import sys

# Read out command line arguments.
#  [1] Port to run server on
port = int(sys.argv[1])

#  [2] Location of list of network descriptions
desc_list_name = sys.argv[2]

#  [3] Location of sample file
source_name = sys.argv[3]

#  [4] Location of indices to use
if len(sys.argv) > 3:
    random_indices_name = sys.argv[4]
else:
    random_idnices_name = None

# LOAD DESCRIPTION FILES
# ======================
desc_files = open(desc_list_name).read().split('\n')[:-1]

descriptions = {}
for desc_file in desc_files:
    print('Loading %s' % (desc_file,))
    loaded = [x.numpy() for x in load_lua(desc_file + '.nc.t7')]

    print('Normalizing...')
    stack = numpy.concatenate(loaded)
    mean = stack.mean(0)
    stack -= mean
    stdev = (stack ** 2).mean(0) ** 0.5

    # Get the "network name", which we take as the filename of the descirption, without
    # directories or extensions.
    descriptions[os.path.split(desc_file)[1].split('.')[0]] = [((x - mean) / stdev).tolist() for x in loaded]
    print('Done.')

# LOAD SOURCE LINES
with open(source_name) as f:
    lines = [line for line in f.read().split('\n')[:-1] if len(line.split(' ')) < 250]

# LOAD RANDOM INDICES
# ===================

# If no random index file was provided,
# generate random indices.
if random_indices_name is None:
    numpy.random.seed(0)
    indices = [numpy.random.randint(len(x.split(' '))) for x in lines]

# Otherwise, use indices provided.
else:
    with open(random_indices_name) as f:
        indices = [index - 1 for index in json.load(f)]

# FORMAT PAGE TO SEND
# ===================

# Format a line with the given index bolded
def index_format(line, index):
    tokens = line.split(' ')
    new_tokens = tokens[:index] + ['<b>' + tokens[index] + '</b>'] + tokens[index + 1:]

    return new_tokens

for i, line in enumerate(lines):
    lines[i] = index_format(line, indices[i])

# Get background color given activation, and return
# span with that background color
def color(activation, token):
    if activation > 0:
        r = 255
        g = 0
        b = 0
        a = 1 - 0.5 ** activation
    elif activation < 0:
        r = 0
        g = 0
        b = 255
        a = 1 - 0.5 ** -activation
    return (
        '<span style="background-color:rgba(%d, %d, %d, %f);" title="%f">' % (r, g, b, a, activation) +
        token +
        '</span>'
    )

class ListServer(BaseHTTPRequestHandler):
    def _set_headers(self, content_type = 'text/html'):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)

        # LENGTH PLOT
        # ===========
        if url.path == '/position-plot.png':
            # Get specified neuron
            query = url.query
            query_components = dict(qc.split('=') for qc in query.split('&'))
            neuron_name = query_components['neuron']

            network_name = neuron_name.split(':')[0]
            neuron_index = int(neuron_name.split(':')[1])

            # Get activations
            activations = [
                vector[indices[i]][neuron_index] for i, vector in enumerate(descriptions[network_name])
            ]

            # Plot activations against indices used
            plt.clf()
            plt.scatter(activations, indices)

            # Send png
            self._set_headers(content_type = 'image/png')
            plt.savefig(self.wfile, format='png')

        # TOKEN-WISE ACTIVATION VISUALIZATION
        # ===================================
        else:
            # Get specified neuron
            query = url.query
            query_components = dict(qc.split('=') for qc in query.split('&'))
            neuron_name = query_components['neuron']

            network_name = neuron_name.split(':')[0]
            neuron_index = int(neuron_name.split(':')[1])

            print(network_name)
            print(neuron_index)

            # Get activations for this neuron over all samples
            activations = [
                vector[indices[i]][neuron_index] for i, vector in enumerate(descriptions[network_name])
            ]

            # Assemble the lines:
            body = '<html><body><table>'

            annotated_lines = []

            # Color all of the lines appropriately and collect them
            for i, line in enumerate(lines):
                annotated_line_broken = [
                    color(descriptions[network_name][i][j][neuron_index], token)
                    for j, token in enumerate(line)
                ]
                annotated_lines.append(' '.join(annotated_line_broken))

            # Sort them by their activations at the given random tokens
            enumerated = [
                (activations[x[0]], x[1]) for x in enumerate(annotated_lines)
            ]

            sorted_lines = sorted(
                enumerated,
                key = lambda x: x[0]
            )

            # Concatenate them together
            for line in sorted_lines:
                body += '<tr><td>%f</td><td>%s</td>' % line

            body += '</table></body></html>'

            # Send the resulting file
            self._set_headers()
            self.wfile.write(body)

# Run the server.
httpd = HTTPServer(
    ('', port),
    ListServer
)

print('Running server on %d' % (port,))

httpd.serve_forever()
