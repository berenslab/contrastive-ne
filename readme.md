# (parametric) (noise-)(contrastive) neighbor embedding methods

Title is work-in-progress.

Some code that can create a parametric embedding suitable for data
visualization.

# Installation

Clone this repository
```sh
git clone https://github.com/berenslab/neighbor-embeddings # name might get changed
cd neighbor-embeddings
pip install .
```

This should install all dependecies and allow the code to be run.
Note that pytorch with GPU support can be a bit pesky to install as a
dependency, so if it's not installed already anyways it might make
sense to consult the [pytorch website](https://pytorch.org) to install
it with CUDA support.

# Example

The most basic use is via `CNE`.  You can create a parametric embedding as follows:

```python
import cne
import numpy as np

# load data
ar = np.load("../contrastive-experiments/data/mnist/pca/data.npy")

emb = cne.CNE()
Y = emb.fit_transform(ar)
```

This will create a `n x 2` numpy array (variable `Y`) which can then be
plotted or further investigated/processed.

## technical detail

The object `ContrastiveEmbedding` needs a neural network `model` as a
required parameter in order to be created.  The `fit` method then
takes a `torch.utils.data.DataLoader` that will be used for training.
The data loader returns a pair of two neighboring points.  In the
classical NE setting this would be two points that share an edge in
the kNN graph; the contrastive learning approach will transform a
sample twice and return those as a “positive edge” which will denote
the attractive force between the two points.
