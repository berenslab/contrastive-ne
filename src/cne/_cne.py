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
    def __init__(self, neighbor_mat, batch_size=1024, shuffle=False, data_on_gpu=False, drop_last=False, seed=0):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :param data_on_gpu: If True, the dataset is loaded on GPU as a whole.
        :param drop_last: Drop the last batch if it is smaller than the others.
        :param seed: Random seed

        :returns: A FastTensorDataLoader.
        """

        neighbor_mat = neighbor_mat.tocoo()
        tensors = [torch.tensor(neighbor_mat.row),
                   torch.tensor(neighbor_mat.col)]
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)

        # manage device
        if data_on_gpu:
            self.device = "cuda"
            tensors = [tensor.to(self.device) for tensor in tensors]
        else:
            self.device = "cpu"
        self.tensors = tensors

        self.dataset_len = torch.tensor(self.tensors[0].shape[0], device=self.device)
        self.batch_size = torch.tensor(batch_size, dtype=int).to(self.device)

        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        torch.manual_seed(self.seed)

        # Calculate number of  batches
        n_batches = torch.div(self.dataset_len, self.batch_size, rounding_mode="floor")
        remainder = torch.remainder(self.dataset_len, self.batch_size)
        if remainder > 0 and not self.drop_last:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.device)
        else:
            self.indices = None
        self.i = torch.tensor(0, device=self.device)
        return self

    def __next__(self):
        if (self.i > self.dataset_len - self.batch_size and self.drop_last) or self.i >= self.dataset_len:
            raise StopIteration

        start = self.i
        end = torch.minimum(self.i + self.batch_size, self.dataset_len)
        if self.indices is not None:
            indices = self.indices[start:end]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[start:end] for t in self.tensors)
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
                 data_on_gpu="auto",
                 use_keops=None,
                 seed=0,
                 anneal_lr=True,
                 embd_dim=2,
                 **kwargs):
        """
        :param model: Embedding model
        :param k: int Number of nearest neighbors
        :param parametric: bool If True and model=None uses a parametric embedding model
        :param data_on_gpu: bool or "auto" Load whole dataset to GPU and try to use pykeops for kNN graph if possible.
        :param use_keops: bool If True use pykeops for kNN graph computation. If False use annoy. Supercedes the kNN
        graph selection by data_on_gpu if not None.
        :param seed: int Random seed
        :param anneal_lr: bool If True anneal the learning rate linearly.
        :param kwargs:
        """
        self.model = model
        self.k = k
        self.parametric = parametric
        self.data_on_gpu = data_on_gpu
        self.use_keops = use_keops
        self.kwargs = kwargs
        self.seed = seed
        self.anneal_lr = anneal_lr
        self.embd_dim = embd_dim


    def fit_transform(self, X, init=None, graph=None):
        "Fit the model, then transform."
        self.fit(X, init=init, graph=graph)
        return self.transform(X, fit_transform=True)

    def transform(self, X, fit_transform=False):
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
            if isinstance(X, int):
                embd = embd[X]
            elif isinstance(X, np.ndarray) and len(np.squeeze(X).shape) == 1:
                if X.dtype == int:
                    embd = embd[np.squeeze(X)]
            elif isinstance(X, list) and np.all([isinstance(x, int) for x in X]):
                embd = embd[X]
            else:
                if not fit_transform:
                    print("Warning: A non-parametric model cannot transform new data. Returning the embedding of the training data. "
                          "Pass an integer (or a list / np.array thereof) to obtain the corresponding training embeddings")
        return embd

    def fit(self, X, init=None, graph=None):
        """
        Fit the model
        :param X: np.array Dataset
        :param init: np.array Initial embedding. If None, use PCA rescaled so that first PC has standard deviation 1.
        :param graph: graph encoding similarity. If None, a kNN graph will be computed. This is done with pykeops if the
        ContrastiveEmbedding instance is on GPU, otherwise annoy is used. Passing "annoy" or "pykeops" forces to use this library for
         the kNN graph computation. Pykeops requires a GPU but is much faster. Alternatively, any scipy.sparse.csr_matrix
         can be passed.
        :return:
        """
        start_time = time.time()
        X = X.reshape(X.shape[0], -1)
        in_dim = X.shape[1]
        # set up model if not given
        if self.model is None:
            if self.parametric:
                self.embd_layer = torch.nn.Embedding.from_pretrained(torch.tensor(X).to(torch.float32),
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
                                        anneal_lr=self.anneal_lr,
                                        **self.kwargs)

        # is no graph is passed, compute the similarity graph with pykeops is cuda is available and otherwise annoy
        if graph is None:
            # select annoy or pykeops depending on data_on_gpu and availability of pykeops
            if self.use_keops is None:
                if self.cne.device == "cuda":
                    try:
                        import pykeops
                        graph = "pykeops"
                    except ImportError:
                        graph = "annoy"
                else:
                    graph = "annoy"
            else:
                graph = "pykeops" if self.use_keops else "annoy"


        if isinstance(graph, str):
            if graph == "annoy":
                print("Computing approximate kNN graph with annoy")
                # create approximate NN search tree
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
            elif graph == "pykeops":
                print("Computing exact kNN graph with pykeops")
                from pykeops.torch import LazyTensor
                import scipy.sparse

                # set up pykeops LazyTensors
                x_cuda = torch.tensor(X).to("cuda").contiguous()
                x_i = LazyTensor(x_cuda[:, None])
                x_j = LazyTensor(x_cuda[None])

                # compute distance and knn_idx with keops
                dists = ((x_i - x_j) ** 2).sum(-1)
                knn_idx = dists.argKmin(K=self.k + 1, dim=0)[:, 1:].cpu().numpy().flatten()

                # construct the adjacency matrix for the graph
                knn_graph = scipy.sparse.coo_matrix((np.ones(len(X) * self.k),
                                                     (np.repeat(np.arange(X.shape[0]), self.k),
                                                      knn_idx)),
                                                    shape=(len(X), len(X)))

                # symmetrize on the fly
                self.neighbor_mat = knn_graph.maximum(knn_graph.transpose()).tocsr()
            else:
                raise ValueError("When passing a string as graph it must be 'annoy' or 'pykeops'")
        else:
            self.neighbor_mat = graph.tocsr()

        if self.data_on_gpu == "auto":
            self.data_on_gpu = self.cne.device == "cuda"
        if self.data_on_gpu and self.cne.device == "cpu":
            print("Warning: Data is on GPU but the model is on CPU. This will be unnecessarily slow.")

        # create data loader
        self.dataloader = FastTensorDataLoader(self.neighbor_mat,
                                               shuffle=True,
                                               batch_size=self.cne.batch_size,
                                               data_on_gpu=self.data_on_gpu,
                                               seed=self.seed)

        # fit the model
        self.cne.fit(self.dataloader, len(X))
        end_time = time.time()
        self.cne.time = end_time - start_time
        return self
