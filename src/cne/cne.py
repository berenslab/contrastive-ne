import sys
import time

import numpy as np
import torch


def train(
    train_loader,
    model,
    log_Z,
    criterion,
    optimizer,
    epoch,
    clip_grad=True,
    print_freq=None,
    force_resample=None,
):
    """
    one epoch training
    :param train_loader: DataLoader Returns batches of similar tuples
    :param model: torch.nn.Module Embedding layer (non-parametric) or neural network (parametric)
    :param log_Z: torch.tensor Float containing the logarithm of the learnable Z
    :param criterion: torch.nn.Module Computes the loss
    :param optimizer:  torch.optim.Optimizer
    :param epoch: int Current training epoch
    :param clip_grad: bool If True, clips gradients to 4
    :param print_freq: int or None Frequency for printing if not None
    :param force_resample: bool or None If True, forces resampling of negative sample indices for every batch. If None, once every epoch.
    :return: torch.tensor Losses for all batches of the epoch
    """
    model.train()
    losses = []

    for idx, (item, neigh) in enumerate(train_loader):
        print_now = print_freq is not None and (idx + 1) % print_freq == 0
        start = time.time()

        images = torch.cat([item, neigh], dim=0)

        images = images.to(next(model.parameters()).device)

        # compute loss
        features = model(images)
        if print_now:
            features.retain_grad()  # to print model agnostic grad statistics
        force_resample = force_resample if force_resample is not None else idx == 0
        loss = criterion(features, log_Z, force_resample=force_resample)

        # update metric
        losses.append(loss.item())

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_value_(model.parameters(), 4)
            if log_Z is not None:
                torch.nn.utils.clip_grad_value_(log_Z, 4)
        optimizer.step()

        # print info
        if print_now:
            print(
                f"Train: E{epoch}, {idx}/{len(train_loader)}\t"
                # print grad on features to be model agnostic
                f"grad magn {features.grad.abs().sum():.3f}, "
                f"loss {sum(losses) / len(losses):.3f}, "
                f"time/iteration {time.time() - start:.3f}",
                file=sys.stderr,
            )
            if torch.isnan(features).any() or torch.isnan(loss).any():
                print(
                    f"NaN error! feat% {torch.isnan(features).sum() / (features.shape[0] * features.shape[1]):.3f}, "
                    f"loss% {torch.isnan(loss).sum():.3f}",
                    file=sys.stderr,
                )
                exit(3)

    return losses


class ContrastiveEmbedding(object):
    """
    Class for computing contrastive embeddings from similarity information.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size=2**15,
        negative_samples=5,
        n_epochs=50,
        device="cuda:0",
        learning_rate=0.001,
        lr_min_factor=0.1,
        momentum=0.9,
        temperature=0.5,
        noise_in_estimator=None,
        neg_spec=None,
        ince_spec=None,
        Z_bar=None,
        s=1.0,
        eps=1.0,
        clamp_high="auto",
        clamp_low="auto",
        Z=1.0,
        loss_mode="infonce",
        metric="euclidean",
        optimizer="adam",
        weight_decay=0,
        anneal_lr="none",
        lr_decay_rate=0.1,
        lr_decay_epochs=None,  # unused for now
        clip_grad=True,
        save_freq=25,
        callback=None,
        print_freq_epoch=None,
        print_freq_in_epoch=None,
        seed=0,
        loss_aggregation="mean",
        force_resample=None,
        warmup_epochs=0,
        warmup_lr=0,
    ):
        """
        :param model: torch.nn.Module Embedding model (embedding layer for non-parametric, neural network for parametric)
        :param batch_size: int Batch size
        :param negative_samples: int Number of negative samples per positive sample
        :param n_epochs: int Number of optimization epochs
        :param device: torch.device Device of optimization
        :param learning_rate: float Learning rate
        :param lr_min_factor: float Minimal value to which learning rate is annealed
        :param momentum: float Momentum of SGD
        :param temperature: float Temperature used in Cosine similarity
        :param noise_in_estimator: float Value used in negative sampling's fraction q / (q+ noise_in_estimator), redundant with Z_bar. Deprecated, use neg_spec instead.
        :param neg_spec: float Value used in negative sampling's fraction q / (q+ neg_spec). Controls the point on the negative sampling spectrum. Redundant with Z_bar, which has priority.
        :param ince_spec: float The repulsive term in the InfoNCE loss is multiplied by the inverse of ince_spec. Controls the InfoNCE spectrum. Redundant with s, which has priority.
        :param Z_bar: float Fixed normalization constant in negative sampling. Controls the negative sampling spectrum. Redundant with s, which has priority.
        :param s: float Slider parameter for the spectrum of neighbor embeddings. If s=1, the embedding will be similar to a UMAP embedding. If s=0, the embedding will be similar to a t-SNE embedding. Logarithmic inter-/ extrapolation. Works for the loss modes "infonce", "infonce_alt" and "neg".
        :param eps: float Iterpolates between UMAP's implicit similarity (eps = 0) and the Cauchy kernels (eps = 1.0)
        :param clamp_high: float Upper value at which arguments to logarithms are clamped. Default "auto" chooses values based on the metric. For metric="euclidean" it is 1.0, for metric="cosine" it is inf.
        :param clamp_low: float Lower value at which arguments to logarithms are clamped. Default "auto" chooses values based on the metric. For metric="euclidean" it is 1e-4, for metric="cosine" it is -inf.
        :param Z: float Initial value for the learned normalization parameter of NCE
        :param loss_mode: str Specifies which loss to use. Must be one of "umap", "neg", "nce", "infonce", "infonce_alt". "neg_sample" is deprecated and defaults to "neg"
        :param metric: str Specifies which metric to use for computing distances. Must be "cosine" or "euclidean".
        :param optimizer: str Specifies which optimizer to use. Must be "sgd" or "adam"
        :param weight_decay: float Value of weight decay.
        :param anneal_lr: bool If True, the learning rate is annealed
        :param lr_decay_rate: float Parameter for speed of learing rate decay
        :param lr_decay_epochs: int Number of epochs over which learning rate is decayed
        :param clip_grad: bool If True, gradients are clipped
        :param save_freq: int Frequency in epochs of calling callback.
        :param callback: callable Callback to call before first and every save_freq epochs.
        :param print_freq_epoch: int Epoch progress is printed every print_freq_epoch epoch
        :param print_freq_in_epoch: int Losses are printed every print_freq_in_epoch batch per epoch
        :param seed: int Random seed
        :param loss_aggregation: str Specifies how to aggregate loss over a batch. Must be "sum" or "mean".
        :param force_resample: bool or None If True, negative sample indices are resampled every batch. If None, they are resampled every epoch.
        :param warmup_epochs: int Number of epochs for linearly warming up the learning rate
        :param warmup_lr: float Starting learning rate to warm up from.
        """
        self.model: torch.nn.Module = model
        self.batch_size: int = batch_size
        self.negative_samples: int = negative_samples
        self.n_epochs: int = n_epochs
        self.device = device
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.temperature = temperature
        self.loss_mode: str = loss_mode
        self.metric: str = metric
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        if isinstance(anneal_lr, bool):
            anneal_lr = "linear" if anneal_lr else "none"
        self.anneal_lr: str = anneal_lr
        self.lr_min_factor: float = lr_min_factor
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_epochs = lr_decay_epochs
        self.warmup_lr = warmup_lr
        self.warmup_epochs = warmup_epochs
        self.clip_grad: bool = clip_grad
        self.save_freq: int = save_freq
        self.callback = callback
        self.print_freq_epoch = print_freq_epoch
        self.print_freq_in_epoch = print_freq_in_epoch
        self.eps = eps
        self.seed = seed
        self.loss_aggregation = loss_aggregation
        self.force_resample = force_resample
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.log_Z = torch.tensor(np.log(Z), device=self.device)
        # alias for loss mode "neg" to ensure backwards compatibility
        # still support "neg_sample" since the loss mode is put into the file names, which are featured in the notebooks
        if self.loss_mode == "neg_sample":
            self.loss_mode = "neg"

        if self.loss_mode == "nce":
            self.log_Z = torch.nn.Parameter(self.log_Z, requires_grad=True)

        if self.loss_mode == "neg":
            n_specified_params = (noise_in_estimator is not None) + (Z_bar is not None) + (s is not None) + (neg_spec is not None)
            assert (
                n_specified_params > 0
                # noise_in_estimator is not None or Z_bar is not None or s is not None
            ), f"Exactly one of 'neg_spec', 'Z_bar', 's', or 'noise_in_estimator' must be not None."

            if n_specified_params > 1:
                print(
                    "Warning: More than one of 'noise_in_estimator', 'neg_spec', 'Z_bar' and "
                    "'s' were specified. 's' will supersede 'Z_bar', which supersedes 'neg_spec', which supersedes 'noise_in_estimator'."
                )

        if self.loss_mode == "infonce" or self.loss_mode == "infonce_alt":
            n_specified_params = (noise_in_estimator is not None) + (ince_spec is not None) + (s is not None)
            assert (
                    n_specified_params > 0
                # noise_in_estimator is not None or Z_bar is not None or s is not None
            ), f"Exactly one of 'noise_in_estimator', 'ince_spec' and 's' must be not None."

            if n_specified_params > 1:
                print(
                    "Warning: More than one of 'noise_in_estimator', 'ince_spec', and "
                    "'s' were specified. 's' will supersede 'ince_spec', which supersedes 'noise_in_estimator'."
                )

        # set up clamping values depending on loss mode
        if clamp_low == "auto":
            if self.metric == "euclidean":
                self.clamp_low = 1e-4
            elif self.metric == "cosine":
                self.clamp_low = float("-inf")
            else:
                raise ValueError(f"Unknown metric {self.metric}")
        else:
            self.clamp_low = clamp_low
        if clamp_high == "auto":
            if self.metric == "euclidean":
                self.clamp_high = 1.0
            elif self.metric == "cosine":
                self.clamp_high = float("inf")
            else:
                raise ValueError(f"Unknown metric {self.metric}")
        else:
            self.clamp_high = clamp_high

        self.s = s
        self.Z_bar = Z_bar
        self.spec_param = noise_in_estimator
        self.neg_spec = neg_spec
        self.ince_spec = ince_spec

        # move to correct device at init, esp before registering with the optimizer
        self.model = self.model.to(self.device)

    def fit(self, X: torch.utils.data.DataLoader, n: int = None):
        """
        Train the model
        :param X: torch.utils.data.DataLoader Loads pairs of similar objects
        :param n: int Size of the dataset
        :return: self
        """
        # translate various spectrum parameters to self.spec_param.
        if self.loss_mode == "neg":
            # if not explicitly passed, use dataset length, only works if DataLoader is over data points, not over similar pairs
            n = len(X) if n is None else n
            if self.s is not None:
                # overwrite self.Z_bar
                Z_umap = n**2 / self.negative_samples
                Z_tsne = 100 * n
                # s = 0 --> z_tsne, s = 1 --> z_umap; using logs for numerical stability
                self.Z_bar = Z_tsne * np.exp(self.s * (np.log(Z_umap) - np.log(Z_tsne)))

            if self.Z_bar is not None:
                # assume uniform noise distribution over n**2 many edges
                self.neg_spec = self.negative_samples * self.Z_bar / n**2

            if self.neg_spec is not None:
                self.spec_param = self.neg_spec

        elif self.loss_mode == "infonce" or self.loss_mode == "infonce_alt":
            if self.s is not None:
                # overwrite self.spec_param, which will be passed to the loss
                # s=0 --> tsne (self.ince_param=1), s=1 --> umap-like (self.ince_param=4); logarithmic interpolation
                self.ince_spec = np.exp(self.s*np.log(4.0))

            # for passing to the loss
            if self.ince_spec is not None:
                self.spec_param = self.ince_spec

        # set up loss
        criterion = ContrastiveLoss(
            negative_samples=self.negative_samples,
            metric=self.metric,
            temperature=self.temperature,
            loss_mode=self.loss_mode,
            spec_param=torch.tensor(self.spec_param).to(self.device),
            eps=torch.tensor(self.eps).to(self.device),
            clamp_high=self.clamp_high,
            clamp_low=self.clamp_low,
            seed=self.seed,
            loss_aggregation=self.loss_aggregation,
        )

        # set up optimizer
        params = [{"params": self.model.parameters()}]
        if self.loss_mode == "nce":
            params += [
                {"params": self.log_Z, "lr": 0.001}
            ]  # make sure log_Z always has a sufficiently small lr

        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params,
                weight_decay=self.weight_decay,
                lr=self.learning_rate,
            )
        else:
            raise ValueError(
                f"Only optimizer 'adam' and 'sgd' allowed, but is {self.optimizer}."
            )

        # initial callback
        if (
            self.save_freq is not None
            and self.save_freq > 0
            and callable(self.callback)
        ):
            self.callback(
                -1,
                self.model,
                self.negative_samples,
                self.loss_mode,
                self.log_Z,
            )

        batch_losses = []

        # logging memory usage
        mem_dict = {
            "active_bytes.all.peak": [],
            "allocated_bytes.all.peak": [],
            "reserved_bytes.all.peak": [],
            "reserved_bytes.all.allocated": [],
        }

        # training
        for epoch in range(self.n_epochs):
            if "cuda" in self.device:
                info = torch.cuda.memory_stats(self.device)
                [mem_dict[k].append(info[k]) for k in mem_dict.keys()]

            # anneal learning rate
            lr = new_lr(
                self.learning_rate,
                self.anneal_lr,
                self.lr_decay_rate,
                lr_min_factor=self.lr_min_factor,
                cur_epoch=epoch,
                total_epochs=self.n_epochs,
                decay_epochs=self.lr_decay_epochs,
                warmup_epochs=self.warmup_epochs,
                warmup_lr=self.warmup_lr,
            )

            # just change the lr of the first param group, not that of Z
            optimizer.param_groups[0]["lr"] = lr

            # train for one epoch
            bl = train(
                X,
                self.model,
                self.log_Z,
                criterion,
                optimizer,
                epoch,
                clip_grad=self.clip_grad,
                print_freq=self.print_freq_in_epoch,
                force_resample=self.force_resample,
            )
            batch_losses.append(bl)

            # callback
            if (
                self.save_freq is not None
                and self.save_freq > 0
                and epoch % self.save_freq == 0
                and callable(self.callback)
            ):
                self.callback(
                    epoch, self.model, self.negative_samples, self.loss_mode, self.log_Z
                )
            # print epoch progress
            if self.print_freq_epoch is not None and epoch % self.print_freq_epoch == 0:
                print(f"Finished epoch {epoch}/{self.n_epochs}", file=sys.stderr)

        self.losses = batch_losses
        self.mem_dict = mem_dict
        self.embedding_ = None
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_


class ContrastiveLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self,
        negative_samples=5,
        temperature=0.07,
        loss_mode="all",
        metric="euclidean",
        base_temperature=1,
        eps=1.0,
        spec_param=1.0,
        clamp_high=1.0,
        clamp_low=1e-4,
        seed=0,
        loss_aggregation="mean",
    ):
        super(ContrastiveLoss, self).__init__()
        self.negative_samples = negative_samples
        self.temperature = temperature
        self.loss_mode = loss_mode
        self.metric = metric
        self.base_temperature = base_temperature
        self.spec_param = spec_param
        self.eps = eps
        self.clamp_high = clamp_high
        self.clamp_low = clamp_low
        self.seed = seed
        torch.manual_seed(self.seed)
        self.neigh_inds = None
        self.loss_aggregation = loss_aggregation

    def forward(self, features, log_Z=None, force_resample=False):
        """Compute loss for model. SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [2 * bsz, n_views, ...].
            log_Z: scalar, logarithm of the learnt normalization constant for nce.
            force_resample: Whether the negative samples should be forcefully resampled.
        Returns:
            A loss scalar.
        """

        batch_size = features.shape[0] // 2
        b = batch_size

        # We can at most sample this many samples from the batch.
        # `b` can be lower than `self.negative_samples` in the last batch.
        negative_samples = min(self.negative_samples, 2 * b - 1)

        if force_resample or self.neigh_inds is None:
            neigh_inds = make_neighbor_indices(
                batch_size, negative_samples, device=features.device
            )
            self.neigh_inds = neigh_inds
        # # untested logic to accomodate for last batch
        # elif self.neigh_inds.shape[0] != batch_size:
        #     neigh_inds = make_neighbor_indices(batch_size, negative_samples)
        #     # don't save this one
        else:
            neigh_inds = self.neigh_inds
        neighbors = features[neigh_inds]

        # `neigh_mask` indicates which samples feel attractive force
        # and which ones repel each other
        neigh_mask = torch.ones_like(neigh_inds, dtype=torch.bool)
        neigh_mask[:, 0] = False

        origs = features[:b]

        # compute probits
        if self.metric == "euclidean":
            dists = ((origs[:, None] - neighbors) ** 2).sum(axis=2)
            # Cauchy affinities
            probits = torch.div(1, self.eps + dists)
        elif self.metric == "cosine":
            norm = torch.nn.functional.normalize
            o = norm(origs.unsqueeze(1), dim=2)
            n = norm(neighbors.transpose(1, 2), dim=1)
            logits = torch.bmm(o, n).squeeze() / self.temperature
            # logits_max, _ = logits.max(dim=1, keepdim=True)
            # logits -= logits_max.detach()
            # logits -= logits.max().detach()
            probits = torch.exp(logits)
        else:
            raise ValueError(f"Unknown metric “{self.metric}”")

        # compute loss
        if self.loss_mode == "nce":
            # for proper nce it should be negative_samples * p_noise. But for
            # uniform noise distribution we would need the size of the dataset
            # here. Also, we do not use a uniform noise distribution as we sample
            # negative samples from the batch.

            if self.metric == "euclidean":
                # estimator is (cauchy / Z) / ( cauchy / Z + neg samples)). For numerical
                # stability rewrite to 1 / ( 1 + (d**2 + eps) * Z * m)
                estimator = 1 / (
                    1 + (dists + self.eps) * torch.exp(log_Z) * negative_samples
                )
            else:
                probits = probits / torch.exp(log_Z)
                estimator = probits / (probits + negative_samples)

            loss = -(~neigh_mask * torch.log(estimator.clamp(self.clamp_low, self.clamp_high))) - (
                neigh_mask * torch.log((1 - estimator).clamp(self.clamp_low, self.clamp_high))
            )
        elif self.loss_mode == "neg":
            if self.metric == "euclidean":
                # estimator rewritten for numerical stability as for nce
                estimator = 1 / (1 + self.spec_param * (dists + self.eps))
            else:
                estimator = probits / (probits + self.spec_param)

            loss = -(~neigh_mask * torch.log(estimator.clamp(self.clamp_low, self.clamp_high))) - (
                neigh_mask * torch.log((1 - estimator).clamp(self.clamp_low, self.clamp_high))
            )

        elif self.loss_mode == "umap":
            # cross entropy parametric umap loss
            loss = -(~neigh_mask * torch.log(probits.clamp(self.clamp_low, self.clamp_high))) - (
                neigh_mask * torch.log((1 - probits).clamp(self.clamp_low, self.clamp_high))
            )
        elif self.loss_mode == "infonce":
            # loss from e.g. sohn et al 2016, includes pos similarity in denominator
            loss = -(self.temperature / self.base_temperature) * (
                (torch.log(probits.clamp(self.clamp_low, self.clamp_high)[~neigh_mask]))
                - self.spec_param**(-1) * torch.log(probits.clamp(self.clamp_low, self.clamp_high).sum(axis=1))
            )
        elif self.loss_mode == "infonce_alt":
            # loss simclr
            loss = -(self.temperature / self.base_temperature) * (
                (torch.log(probits.clamp(self.clamp_low, self.clamp_high)[~neigh_mask]))
                - self.spec_param**(-1) * torch.log((neigh_mask *
                                                       probits.clamp(self.clamp_low, self.clamp_high)).sum(axis=1))
            )
        else:
            raise ValueError(f"Unknown loss_mode “{self.loss_mode}”")

        # aggregate loss over batch
        if self.loss_aggregation == "sum":
            loss = loss.sum()
        else:
            loss = loss.mean()

        return loss


def new_lr(
    learning_rate,
    anneal_lr,
    lr_decay_rate,
    lr_min_factor,
    cur_epoch,
    total_epochs,
    decay_epochs=None,  # unused for now
    warmup_lr=0,
    warmup_epochs=0,
):
    """
    Decays the learning rate
    :param learning_rate: float Current learning rate
    :param anneal_lr: str Specifies the learning rate annealing. Must be one of "none", "linear" or "cosine"
    :param lr_decay_rate: float Rate of cosine decay.
    :param lr_min_factor: float Minimal learning rate of linear decay.
    :param cur_epoch: int Current epoch
    :param total_epochs: int Total number of epochs
    :param decay_epochs: int Number of decay epochs (unused)
    :param warmup_epochs: int Number of epochs for linearly warming up the learning rate
    :param warmup_lr: float Starting learning rate to warm up from.
    :return: float New learning rate
    """
    anneal_epochs = total_epochs - warmup_epochs
    if cur_epoch < warmup_epochs:
        lr = warmup_lr + (learning_rate - warmup_lr) * cur_epoch / warmup_epochs
    else:
        cur_epoch = cur_epoch - warmup_epochs
        if anneal_lr == "none":
            lr = learning_rate
        elif anneal_lr == "linear":
            lr = learning_rate * max(lr_min_factor, 1 - cur_epoch / anneal_epochs)
        elif anneal_lr == "cosine":
            eta_min = 0
            lr = (
                eta_min
                + (learning_rate - eta_min)
                * (1 + np.cos(np.pi * cur_epoch / anneal_epochs))
                / 2
            )
        else:
            raise RuntimeError(f"Unknown learning rate annealing “{anneal_lr = }”")

    return lr


def make_neighbor_indices(batch_size, negative_samples, device=None):
    """
    Selects neighbor indices
    :param batch_size: int Batch size
    :param negative_samples: int Number of negative samples
    :param device: torch.device Device of the model
    :return: torch.tensor Neighbor indices
    :rtype:
    """
    b = batch_size

    if negative_samples < 2 * b - 2:
        # uniform probability for all points in the minibatch,
        # we sample points for repulsion randomly
        neg_inds = torch.randint(0, 2 * b - 1, (b, negative_samples), device=device)
        neg_inds += (torch.arange(1, b + 1, device=device) - 2 * b)[:, None]
    else:
        # full batch repulsion
        all_inds1 = torch.repeat_interleave(
            torch.arange(b, device=device)[None, :], b, dim=0
        )
        not_self = ~torch.eye(b, dtype=bool, device=device)
        neg_inds1 = all_inds1[not_self].reshape(b, b - 1)

        all_inds2 = torch.repeat_interleave(
            torch.arange(b, 2 * b, device=device)[None, :], b, dim=0
        )
        neg_inds2 = all_inds2[not_self].reshape(b, b - 1)
        neg_inds = torch.hstack((neg_inds1, neg_inds2))

    # now add transformed explicitly
    neigh_inds = torch.hstack(
        (torch.arange(b, 2 * b, device=device)[:, None], neg_inds)
    )

    return neigh_inds
