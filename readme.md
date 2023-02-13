# Contrastive Neighbor Embedding Methods

This repository contains code to create parametric and nonparametric embeddings suitable for data
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
similarity (`metric="cosine"`). Its `forward` method accepts a dataloader. If the dataloader implements data augmentation 
one obtains SimCLR. 

## Installation

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

## Example

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
embedder_neg = cne.CNE(loss_mode="neg",
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


## Negative sampling spectrum
Using the negative sampling loss (`loss_mode="neg"`), one can obtain a spectrum of embeddings that includes embeddings
similar to t-SNE and UMAP. It implements a trade-off between preserving discrete (local) and continuous (global) structure. The 
optional arguments `Z_bar` and `s` control the location 
on the spectrum. For both, larger values yield to more attraction and thus better global structure preservation, while 
smaller values lead to a focus on the local structure. 

The two parameters differ in their scaling. The hyperparameter `Z_bar` directly sets the 
fixed normalization constant. This give more direct control, but also requires some knowledge about the suitable range 
for `Z_bar`. In contrast, the 'slider' hyperparameter `s` is more intuitive. Setting `s=0` yields and embedding with 
normalization similar to that of t-SNE (the code set `Z_bar = 100 * n` where `n` is the sample size), 
while setting `s=1` yields an embedding with UMAP's default fixed normalization constant (`Z_bar = n**2 / m` where `m`
is the number of negative samples). Other values for `s` inter- and extrapolate these two special cases 
and thus lead to an embedding between
or beyond t-SNE and UMAP. The default corresponds to `s=1`. If both hyperparameters are set, `s` overrides `Z_bar`.


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
setting on GPU becomes competitive with the reference implementations of UMAP[^umap] and t-SNE[^tsne].

<img width="600" alt="Run times by batch size" src="/figures/runtime_bs.png">


[^umap]: McInnes, Leland, John Healy, and James Melville. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv preprint arXiv:1802.03426 (2018).  
[^nce]: Gutmann, Michael U., and Aapo Hyvärinen. "Noise-Contrastive Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics." Journal of Machine Learning Research 13.2 (2012).  
[^neg]: Mikolov, Tomas, et al. "Distributed Representations of Words and Phrases and their Compositionality." Advances in Neural Information Processing Systems 26 (2013).  
[^infonce]: Oord, Aaron van den, Yazhe Li, and Oriol Vinyals. "Representation Learning with Contrastive Predictive Coding." arXiv preprint arXiv:1807.03748 (2018).  
[^pumap]: Sainburg, Tim, Leland McInnes, and Timothy Q. Gentner. "Parametric UMAP Embeddings for Representation and Semisupervised Learning." Neural Computation 33.11 (2021): 2881-2907.  
[^ncvis]: Artemenkov, Aleksandr, and Maxim Panov. "NCVis: Noise Contrastive Approach for Scalable Visualization." Proceedings of The Web Conference 2020. 2020.  
[^simclr]: Chen, Ting, et al. "A Simple Framework for Contrastive Learning of Visual Representations." International conference on machine learning. PMLR, 2020.  
[^tsne]: Poličar, Pavlin G., Martin Stražar, and Blaž Zupan. "openTSNE: a modular Python library for t-SNE dimensionality reduction and embedding." BioRxiv (2019): 731877.
