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

# plot
plt.figure()
plt.scatter(*embd_ncvis.T, c=y, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title("Parametric NCVis of MNIST")
plt.show()



# non-parametric Neg-t-SNE
embedder_neg = cne.CNE(loss_mode="neg",
                       k=15,
                       optimizer="sgd",
                       parametric=False,
                       print_freq_epoch=10)
embd_neg = embedder_neg.fit_transform(x)

plt.figure()
plt.scatter(*embd_neg.T, c=y, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title(r"Neg-$t$-SNE of MNIST")
plt.show()


# compute spectrum with negative sampling loss
spec_params = [0.0, 0.5, 1.0]

neg_embeddings = {}
graph = None  # for storing the kNN graph once it has been computed
for s in spec_params:
    embedder = cne.CNE(loss_mode="neg",
                       optimizer="sgd",
                       parametric=False,
                       s=s,
                       print_freq_epoch=10)
    embd = embedder.fit_transform(x, graph=graph)
    neg_embeddings[s] = embd

    if graph is None:
        graph = embedder.neighbor_mat

# plot embeddings
fig, ax = plt.subplots(1, len(spec_params), figsize=(5.5, 3), constrained_layout=True)
for i, s in enumerate(spec_params):
    ax[i].scatter(*neg_embeddings[s].T, c=y, alpha=0.5, s=0.1, cmap="tab10", edgecolor="none")
    ax[i].set_aspect("equal", "datalim")
    ax[i].axis("off")
    ax[i].set_title(f"s={s}")

fig.suptitle("Negative sampling spectrum of MNIST")
plt.show()


# compute spectrum with InfoNCE loss
spec_params = [0.0, 0.5, 1.0]

ince_embeddings = {}
for s in spec_params:
    embedder = cne.CNE(loss_mode="infonce",
                       k=15,
                       negative_samples=500,  # higher number of negative samples for higher quality embedding
                       optimizer="sgd",
                       parametric=False,
                       s=s,
                       print_freq_epoch=100)
    embd = embedder.fit_transform(x, graph=graph)
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