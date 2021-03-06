# Contrastive Neighbor Embedding Methods

This repo contains code to create a (non-) parametric embedding suitable for data
visualization with various contrastive losses.

# Installation

Clone this repository
```sh
git clone https://github.com/berenslab/contrastive-ne
cd contrastive-ne
pip install .
```

This installs all dependecies and allows the code to be run.
Note that pytorch with GPU support can be a bit tricky to install as a
dependency, so if it is not installed already, it might make
sense to consult the [pytorch website](https://pytorch.org) to install
it with CUDA support.

# Example

The most basic use is via `CNE`. You can create embeddings as follows:

```python
import cne
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# load MNIST
mnist_train = torchvision.datasets.MNIST(train=True,
                                         download=True, 
                                         transform=None)
x_train, y_train = mnist_train.data.float().numpy(), mnist_train.targets

mnist_test = torchvision.datasets.MNIST(train=False,
                                        download=True, 
                                        transform=None)
x_test, y_test = mnist_test.data.float().numpy(), mnist_test.targets

x = np.concatenate([x_train, x_test], axis=0)
x = x.reshape(x.shape[0], -1)
y = np.concatenate([y_train, y_test], axis=0)

# parametric NCVis 
embedder_ncvis = cne.CNE(loss_mode="nce",
                         k=15,
                         optimizer="adam",
                         parametric=True,
                         print_freq_epoch=10)
embd_ncvis = embedder_ncvis.fit_transform(x)

# non-parametric Neg-t-SNE
embedder_neg = cne.CNE(loss_mode="neg_sample",
                       k=15,
                       optimizer="sgd",
                       momentum=0.0,
                       parametric=False,
                       print_freq_epoch=10)
embd_neg = embedder_neg.fit_transform(x)

# plot embeddings
plt.figure()
plt.scatter(*embd_ncvis.T, c=y, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title("Parametric NCVis of MNIST")
plt.show()
```
<img width="400" alt="Parametric NCVis plot" src="/figures/parametric_ncvis_mnist.png">

```python
plt.figure()
plt.scatter(*embd_neg.T, c=y, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title(r"Neg-$t$-SNE of MNIST")
plt.show()
```
<img width="400" alt="Neg-t-SNE plot" src="/figures/negtsne_mnist.png">

## Technical detail

The object `ContrastiveEmbedding` needs a neural network `model` as a
required parameter in order to be created.  The `fit` method then
takes a `torch.utils.data.DataLoader` that will be used for training.
The data loader returns a pair of two neighboring points.  In the
classical NE setting this would be two points that share an edge in
the kNN graph; the contrastive self-supervised learning approach will transform a
sample twice and return those as a ???positive edge??? which will denote
the attractive force between the two points.
