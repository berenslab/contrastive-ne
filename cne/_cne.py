import torch
import numpy as np

from .cne import ContrastiveEmbedding
from annoy import AnnoyIndex
from scipy.sparse import lil_matrix
from sklearn.decomposition import PCA
import time

# various datasets / dataloaders
class NeighborTransformData(torch.utils.data.Dataset):
    """Returns a pair of neighboring points in the dataset."""
    def __init__(
            self, dataset, neighbor_mat, random_state=None
    ):
        self.dataset = torch.tensor(dataset, dtype=torch.float32)
        self.neighbor_mat = neighbor_mat
        self.rng = np.random.default_rng(random_state)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        neighs = self.neighbor_mat[i].nonzero()[1]
        nidx = self.rng.choice(neighs)

        item = self.dataset[i]
        neigh = self.dataset[nidx]
        return item, neigh


class NeighborTransformIndices(torch.utils.data.Dataset):
    """Returns a pair of indices of neighboring points in the dataset."""
    def __init__(
            self, neighbor_mat, random_state=None
    ):
        neighbor_mat = neighbor_mat.tocoo()
        self.heads = torch.tensor(neighbor_mat.row)
        self.tails = torch.tensor(neighbor_mat.col)

    def __len__(self):
        return len(self.heads)

    def __getitem__(self, i):
        return self.heads[i], self.tails[i]



class NumpyToTensorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, reshape=None):
        self.dataset = torch.tensor(dataset, dtype=torch.float32)
        if reshape is not None:
            self.reshape = lambda x: np.reshape(x, reshape)
        else:
            self.reshape = lambda x: x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        return self.reshape(item)


class NumpyToIndicesDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


# based on https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """
    def __init__(self, neighbor_mat, batch_size=1024, shuffle=False, on_gpu=False, drop_last=False, seed=0):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :param on_gpu: If True, the dataset is loaded on GPU as a whole.
        :param drop_last: Drop the last batch if it is smaller than the others.
        :param seed: Random seed

        :returns: A FastTensorDataLoader.
        """

        neighbor_mat = neighbor_mat.tocoo()
        tensors = [torch.tensor(neighbor_mat.row),
                   torch.tensor(neighbor_mat.col)]
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)

        # manage device
        self.device = "cpu"
        if on_gpu:
            self.device = "cuda"
            tensors = [tensor.to(self.device) for tensor in tensors]
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        torch.manual_seed(self.seed)

        # Calculate number of  batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0 and not self.drop_last:
            n_batches += 1
        self.n_batches = n_batches

        self.batch_size = torch.tensor(self.batch_size, dtype=int).to(self.device)

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i > self.dataset_len - self.batch_size:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class FCNetwork(torch.nn.Module):
    "Fully-connected network"
    def __init__(self, in_dim=784, feat_dim=2):
        super(FCNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, feat_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class CNE(object):
    """
    Manages contrastive neighbor embeddings.
    """
    def __init__(self,
                 model=None,
                 k=15,
                 parametric=False,
                 on_gpu=True,
                 seed=0,
                 loss_aggregation="sum",
                 anneal_lr=True,
                 embd_dim=2,
                 **kwargs):
        """
        :param model: Embedding model
        :param k: int Number of nearest neighbors
        :param parametric: bool If True and model=None uses a parametric embedding model
        :param on_gpu: bool Load whole dataset to GPU
        :param seed: int Random seed
        :param loss_aggregation: str If 'mean' uses mean aggregation of loss over batch, if 'sum' uses sum.
        :param anneal_lr: bool If True anneal the learning rate linearly.
        :param kwargs:
        """
        self.model = model
        self.k = k
        self.parametric = parametric
        self.on_gpu = on_gpu
        self.kwargs = kwargs
        self.seed = seed
        self.loss_aggregation = loss_aggregation
        self.anneal_lr = anneal_lr
        self.embd_dim = embd_dim


    def fit_transform(self, X, init=None, graph=None):
        "Fit the model, then transform."
        self.fit(X, init=init, graph=graph)
        return self.transform(X)

    def transform(self, X):
        "Transform a dataset using the fitted model."
        if self.parametric:
            X = X.reshape(X.shape[0], -1)
            self.dataset_plain = NumpyToTensorDataset(X)
            self.dl_unshuf = torch.utils.data.DataLoader(
                self.dataset_plain,
                shuffle=False,
                batch_size=self.cne.batch_size,
            )
            model = self.network
            device = self.cne.device
            embd = np.vstack([model(batch.to(device))
                            .detach().cpu().numpy()
                            for batch in self.dl_unshuf])
        else:
            embd = self.model.weight.detach().cpu().numpy()


        return embd

    def fit(self, X, init=None, graph=None):
        "Fit the model."
        start_time = time.time()
        X = X.reshape(X.shape[0], -1)
        in_dim = X.shape[1]
        # set up model if not given
        if self.model is None:
            if self.parametric:
                self.embd_layer = torch.nn.Embedding.from_pretrained(torch.tensor(X),
                                                                     freeze=True)
                self.network = FCNetwork(in_dim, feat_dim=self.embd_dim)
                self.model = torch.nn.Sequential(
                    self.embd_layer,
                    self.network
                )
            else:
                if init is None:
                    # default to pca
                    pca_projector = PCA(n_components=self.embd_dim)
                    init = pca_projector.fit_transform(X)
                    init /= (init[:, 0].std())
                elif type(init).__module__ == np.__name__:
                    assert len(init) == len(X), f"Data and initialization must have the same number of elements but have {len(X)} and {len(init)}."
                    assert len(init.shape) == 2, f"Initialization must have 2 dimensions but has {len(init.shape)}."
                    if init.shape[1] != self.embd_dim:
                        print(f"Warning: Initialization has {init.shape[1]} dimensions but {self.embd_dim} are expected."
                              f" Will use the initialization's {init.shape[1]} dimensions.")
                # All embedding parameters will be part of the model. This is
                # conceptually easy, but limits us to embeddings that fit on the
                # GPU.
                self.model = torch.nn.Embedding.from_pretrained(torch.tensor(init))
                self.model.requires_grad_(True)

        # use higher learning rate for non-parametric version
        if "learning_rate" not in self.kwargs.keys():
            lr = 0.001 if self.parametric else 1.0
            self.kwargs["learning_rate"] = lr

        # Load embedding engine
        self.cne = ContrastiveEmbedding(self.model,
                                        seed=self.seed,
                                        loss_aggregation=self.loss_aggregation,
                                        anneal_lr=self.anneal_lr,
                                        **self.kwargs)

        # compute the similarity graph with annoy if none is given
        if graph is None:
            # create approximate NN search tree
            print("Computing approximate kNN graph")
            self.annoy = AnnoyIndex(in_dim, "euclidean")
            [self.annoy.add_item(i, x) for i, x in enumerate(X)]
            self.annoy.build(50)

            # construct the adjacency matrix for the graph
            adj = lil_matrix((X.shape[0], X.shape[0]))
            for i in range(X.shape[0]):
                neighs_, _ = self.annoy.get_nns_by_item(i, self.k + 1, include_distances=True)
                neighs = neighs_[1:]
                adj[i, neighs] = 1
                adj[neighs, i] = 1  # symmetrize on the fly

            self.neighbor_mat = adj.tocsr()
        else:
            self.neighbor_mat = graph.tocsr()

        # create data loader
        self.dataloader = FastTensorDataLoader(self.neighbor_mat,
                                               shuffle=True,
                                               batch_size=self.cne.batch_size,
                                               on_gpu=self.on_gpu,
                                               seed=self.seed)
        # fit the model
        self.cne.fit(self.dataloader, len(X))
        end_time = time.time()
        self.cne.time = end_time - start_time
        return self
