![PyPI - Version](https://img.shields.io/pypi/v/contrastive-ne)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Contrastive Neighbor Embeddings


Parametric and nonparametric neighbor embeddings suitable for data
visualization with various contrastive losses. 

Reference:

* [From t-SNE to UMAP with contrastive learning](https://openreview.net/forum?id=B8a1FcY0vi), _ICLR 2023_  
  Sebastian Damrich, Niklas Böhm, Fred A Hamprecht, Dmitry Kobak
  
```
@inproceedings{damrich2023from,
  title={From $t$-{SNE} to {UMAP} with contrastive learning},
  author={Damrich, Sebastian and B{\"o}hm, Jan Niklas  and Hamprecht, Fred A and Kobak, Dmitry},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```

This repository provides our PyTorch library. The code that implements the specific analysis presented in the paper is available at https://github.com/hci-unihd/cl-tsne-umap.


## Scope

This repository allows to use several different losses, training modes, devices, and distance measures. 
It (re-)implements the UMAP loss[^umap], the negative sampling loss (NEG)[^neg], noise-contrastive estimation loss (NCE)[^nce], and 
InfoNCE loss[^infonce] in PyTorch. All of these losses can be combined with embedding similarities either based on the Cauchy distribution (default) 
or on the cosine distance. The embedding positions can either be optimized directly (non-parametric mode) or a neural network 
can be trained to map data to embedding positions (parametric mode). Our pure PyTorch implementation can run seamlessly on CPU or GPU.

As a result, this library re-implements several existing contrastive methods, alongside many new ones. The most important ones
are summarized the table below.

| Loss              | Non-parametric    | Parametric                     |
|-------------------|-------------------|--------------------------------|
| UMAP[^umap]       | UMAP[^umap]       | Parametric UMAP[^pumap]        |
| NEG[^neg]         | Neg-t-SNE (new)   | Parametric Neg-t-SNE (new)     |
| NCE[^nce]         | NCVis[^ncvis]     | Parametric NCVis (new)         |
| InfoNCE[^infonce] | InfoNC-t-SNE (new) | Parametric InfoNC-t-SNE (new) |



The repository can also be used to run  SimCLR[^simclr] experiments, by using the InfoNCE loss.  The main class 
`ContrastiveEmbedding` allows to change the similarity measure to the exponential of a temperature-scaled cosine 
similarity (`metric="cosine"`). Its `forward` method accepts a dataloader. If the dataloader implements data augmentation, 
one obtains SimCLR. 

## Installation
Pip installation:
```sh
pip install contrastive-ne
```

To install from source, clone this repository
```sh
git clone https://github.com/berenslab/contrastive-ne
cd contrastive-ne
pip install .
```

This installs all dependecies and allows the code to be run.
Note that pytorch with GPU support can be a bit tricky to install as a
dependency, so if it is not installed already, it might make
sense to consult the [pytorch website](https://pytorch.org) to install
it with CUDA support prior to the installation of `contrastive-ne`.

## Example

The most basic usage is via the `CNE` class. Here are some Hello World examples using the MNIST dataset.

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

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)
```

By default, `CNE` uses the InfoNCE loss and thus approximates $t$-SNE (one can increase the number of negative samples, e.g. `negative_samples=500` to get a better approximation, see also below):

```python
# default CNE (using the InfoNCE loss)
embedder = cne.CNE()
embd = embedder.fit_transform(x)

plt.figure()
plt.scatter(*embd.T, c=y, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title(r"InfoNC-$t$-SNE of MNIST")
plt.show()
```

<p align="center"><img width="400" alt="InfoNCE plot" src="/figures/default_mnist.png">

To get non-parametric Neg-t-SNE (very close to UMAP) use `loss_mode="neg"`:

```python
# non-parametric Neg-t-SNE
embedder_neg = cne.CNE(loss_mode="neg")
embd_neg = embedder_neg.fit_transform(x)

plt.figure()
plt.scatter(*embd_neg.T, c=y, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title(r"Neg-$t$-SNE of MNIST")
plt.show()
```
<p align="center"><img width="400" alt="Neg-t-SNE plot" src="/figures/negtsne_mnist.png">

Here is a parametric NCVis (NC-t-SNE) example, highlighting that new embedding points can be added with a parametric embedding:

```python
# parametric NCVis, highlighting the embedding of new points
embedder_ncvis = cne.CNE(loss_mode="nce",
                         optimizer="adam",  # Adam tends to work better than SGD for parametric runs
                         parametric=True)
embd_ncvis_train = embedder_ncvis.fit_transform(x_train)  # only train with training set
embd_ncvis_test = embedder_ncvis.transform(x_test)  # transform test set with the trained model

# plot
titles = ["Train", "Test"]
fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.5), constrained_layout=True)
ax[0].scatter(*embd_ncvis_train.T, c=y_train, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
ax[1].scatter(*embd_ncvis_test.T, c=y_test, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")

for i in range(2):
    ax[i].set_title(titles[i])
    ax[i].set_aspect("equal", "datalim")
    ax[i].axis("off")

fig.suptitle("Parametric NCVis of MNIST")
plt.show()
```
<p align="center"><img width="600" alt="Parametric NCVis plot" src="/figures/parametric_ncvis_mnist.png">


To compute the spectrum of neighbor embeddings with the negative sampling loss, we can use the following code:

```python
# compute spectrum with negative sampling loss
spec_params = [0.0, 0.5, 1.0]

neg_embeddings = {}
for s in spec_params:
    embedder = cne.CNE(loss_mode="neg",
                       s=s)
    embd = embedder.fit_transform(x)
    neg_embeddings[s] = embd

# plot embeddings
fig, ax = plt.subplots(1, len(spec_params), figsize=(5.5, 3), constrained_layout=True)
for i, s in enumerate(spec_params):
    ax[i].scatter(*neg_embeddings[s].T, c=y, alpha=0.5, s=0.1, cmap="tab10", edgecolor="none")
    ax[i].set_aspect("equal", "datalim")
    ax[i].axis("off")
    ax[i].set_title(f"s={s}")

fig.suptitle("Negative sampling spectrum of MNIST")
plt.show()
```

<p align="center"><img width="600" alt="Neg-t-SNE spectrum" src="/figures/neg_spectrum_mnist.png">

A similar spectrum can be computed using the InfoNCE loss (note that this is not described in our paper but was
implemented after it has been published). Using a higher number of negative samples leads to better local structure:
```python
# compute spectrum with InfoNCE loss
spec_params = [0.0, 0.5, 1.0]

ince_embeddings = {}
for s in spec_params:
    embedder = cne.CNE(negative_samples=500,  # more negative samples for better local quality
                       s=s)
    embd = embedder.fit_transform(x)
    ince_embeddings[s] = embd

# plot embeddings
fig, ax = plt.subplots(1, len(spec_params), figsize=(5.5, 3), constrained_layout=True)
for i, s in enumerate(spec_params):
    ax[i].scatter(*ince_embeddings[s].T, c=y, alpha=0.5, s=0.1, cmap="tab10", edgecolor="none")
    ax[i].set_aspect("equal", "datalim")
    ax[i].axis("off")
    ax[i].set_title(f"s={s}")

fig.suptitle("InfoNCE spectrum of MNIST")
plt.show()
```
<p align="center"><img width="600" alt="InfoNC-t-SNE spectrum" src="/figures/infonce_spectrum_mnist_m_500.png">



## Contrastive neighbor embedding spectra
The `CNE` class takes an argument `s` which indicates the position of the embedding on the attraction-repulsion spectrum, 
where `s=0` corresponds to a t-SNE-like embedding and `s=1` corresponds to a UMAP-like embedding. The spectrum is 
implemented for the negative sampling loss mode (`loss_mode=neg`) and the InfoNCE loss mode (`loss_mode=infonce`). It
implements a trade-off between preserving discrete (local) and continuous (global) structure.

For a more fine-grained control, there are arguments specific to the loss mode. For negative sampling one can specify 
the argument `Z_bar` which directly sets the fixed normalization constant or `neg_spec` which sets the value in the denominator 
of the negative sampling estimator. Their relation is `Z_bar = m * neg_spec / n**2` where `n` is the sample size and `m` 
the number of negative samples. The high-level argument `s` corresponds to `Z_bar = 100 * n` for `s=0` and `Z_bar = n**2 / m` 
for `s=1`. Note that the `s=1` value is exactly what UMAP uses, whereas the `s=0` value is our heuristic that usually approximates t-SNE well.

For InfoNCE, the argument `ince_spec` controls the exaggeration of attraction over repulsion. The setting `s=0` corresponds
to `ince_spec=1` and `s=1` corresponds to `ince_spec=4.0`. Here `s=0` recovers t-SNE value exactly, whereas `s=1` is our heuristic
that usually approximates UMAP well.

For all input arguments controlling the position on the spectrum, larger values yield more attraction and thus better global 
structure preservation, while smaller values lead to a focus on the local structure.

## Early exaggeration
Similar to early exaggeration in t-SNE the loss modes `neg` and `infonce` use early exaggeration by default. To disable
it pass `early_exaggeration=False` to the `CNE` class. If used, the first third of the optimization epochs use the setting `s=1.0`
unless a higher value `s` is specified. This is to ensure that the embedding is initialized with a good global structure. 

## Technical details

The object `ContrastiveEmbedding` needs a neural network `model` as a
required parameter in order to be created.  The `fit` method then
takes a `torch.utils.data.DataLoader` that will be used for training.
The data loader returns a pair of two neighboring points.  In the
classical NE setting this would be two points that share an edge in
the kNN graph; the contrastive self-supervised learning approach will transform a
sample twice and return those as a “positive edge” which will denote
the attractive force between the two points.

## Run time analysis
The run time depends strongly on the training mode (parametric / non-parametric), the device (CPU / GPU) and on the 
batch size. The plot below illustrates this for the optimization of a Neg-t-SNE embedding of MNIST. Note that the non-parametric
setting on GPU becomes competitive with the reference (CPU) implementations of UMAP[^umap] and t-SNE[^tsne].

<img width="600" alt="Run times by batch size" src="/figures/runtime_bs.png">


## Logging
There are callbacks for logging the embeddings and losses during training. Note that the loss logging depends on the
[vis_utils](https://github.com/sdamrich/vis_utils) repository, which needs to be installed separately.

[^umap]: McInnes, Leland, John Healy, and James Melville. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv preprint arXiv:1802.03426 (2018).  
[^nce]: Gutmann, Michael U., and Aapo Hyvärinen. "Noise-Contrastive Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics." Journal of Machine Learning Research 13.2 (2012).  
[^neg]: Mikolov, Tomas, et al. "Distributed Representations of Words and Phrases and their Compositionality." Advances in Neural Information Processing Systems 26 (2013).  
[^infonce]: Oord, Aaron van den, Yazhe Li, and Oriol Vinyals. "Representation Learning with Contrastive Predictive Coding." arXiv preprint arXiv:1807.03748 (2018).  
[^pumap]: Sainburg, Tim, Leland McInnes, and Timothy Q. Gentner. "Parametric UMAP Embeddings for Representation and Semisupervised Learning." Neural Computation 33.11 (2021): 2881-2907.  
[^ncvis]: Artemenkov, Aleksandr, and Maxim Panov. "NCVis: Noise Contrastive Approach for Scalable Visualization." Proceedings of The Web Conference 2020. 2020.  
[^simclr]: Chen, Ting, et al. "A Simple Framework for Contrastive Learning of Visual Representations." International conference on machine learning. PMLR, 2020.  
[^tsne]: Poličar, Pavlin G., Martin Stražar, and Blaž Zupan. "openTSNE: a modular Python library for t-SNE dimensionality reduction and embedding." BioRxiv (2019): 731877.
