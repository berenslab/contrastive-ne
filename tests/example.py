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


# non-parametric Neg-t-SNE
embedder_neg = cne.CNE(loss_mode="neg")
embd_neg = embedder_neg.fit_transform(x)
graph = embedder_neg.neighbor_mat  # to avoid the recomputation of the sknn graph below

plt.figure()
plt.scatter(*embd_neg.T, c=y, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title(r"Neg-$t$-SNE of MNIST")
plt.show()

# parametric NCVis
embedder_ncvis = cne.CNE(loss_mode="nce",
                         optimizer="adam",
                         parametric=True)
embd_ncvis = embedder_ncvis.fit_transform(x, graph=graph)

# plot
plt.figure()
plt.scatter(*embd_ncvis.T, c=y, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title("Parametric NCVis of MNIST")
plt.show()


# compute spectrum with negative sampling loss
spec_params = [0.0, 0.5, 1.0]

neg_embeddings = {}
for s in spec_params:
    embedder = cne.CNE(loss_mode="neg",
                       s=s)
    embd = embedder.fit_transform(x, graph=graph)
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


# compute spectrum with InfoNCE loss
spec_params = [0.0, 0.5, 1.0]

ince_embeddings = {}

for s in spec_params:
    embedder = cne.CNE(loss_mode="infonce",
                       negative_samples=500,
                       s=s)
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