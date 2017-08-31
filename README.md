
Visualization Tools for NMT
===========================

This repository contains tools to visualize neurons from [seq2seq-attn](https://github.com/harvardnlp/seq2seq-attn) models. It uses description files generated from [this repository](https://github.com/dabbler0/nmt-shared-information). To use these scripts, you must first generate a description file using `describe.lua` from [nmt-shared-information](https://github.com/dabbler0/nmt-shared-information).

Once you have done so, make a file containing a list of all the description files you want to include in your visualization. This could be just one. For instance, in `desc_list.txt`, put:

```
/path/to/description-1.t7
/path/to/description-2.t7
```

Activation Visualization and Position Plots
--------------------------------------------

Once you have done so, you can visualize individual token activations using `server.py`:

```bash
pythons server.py 8080 desc_list.txt /path/to/sample/file.tok
```

The sample file must be the same one you used with `describe.lua`. This will run a server on the given port (here `8080`). You can then visit either `localhost:8080/?neuron=description-1:123` to get a token visualization of neuron `123` from the network described the file `description-1.t7`, or create a position vs. activation plot with `localhost:8080/position-plot.png?neuron=description-1:123`.

Hierarchical Clustering Dendrogram
-----------------------------------

To cluster the neurons with hierarchical clustering and correlation, run:

```
python clustering.py desc_list.txt output.svg
```

This will write the dendrogram as an svg file to `output.svg`.

Identifying Position Neurons
----------------------------

To identify position neurons using the MSE of a position-to-activation predictor, run:

```
python position-neurons.py /path/to/description-1.t7
```

Note that this is not a path to a description list, but to a description itself.
