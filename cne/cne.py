import sys
import time

import torch


def train(train_loader,
          model,
          log_Z,
          criterion,
          optimizer,
          epoch,
          clip_grad=True,
          print_freq=None):
    """one epoch training"""
    model.train()
    losses = []

    for idx, (item, neigh) in enumerate(train_loader):
        start = time.time()

        images = torch.cat([item, neigh], dim=0)
        # labels = torch.cat([labels[0], labels[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            # labels = labels.cuda(non_blocking=True)

        # compute loss
        features = model(images)
        loss = criterion(features, log_Z)

        # update metric
        # losses.update(loss.item(), bsz)
        losses.append(loss.item())

        # SGD
        optimizer.zero_grad()
        # print(torch.isnan(features).any(), torch.isnan(loss).any(), file=sys.stderr)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_value_(model.parameters(), 4)
            if log_Z is not None:
                torch.nn.utils.clip_grad_value_(log_Z, 4)
        optimizer.step()

        # print info
        if print_freq is not None and (idx + 1) % print_freq == 0:
            print(f'Train: E{epoch}, {idx}/{len(train_loader)}\t'
                  f'grad magn {model.linear_relu_stack[-1].weight.grad.abs().sum()}, loss {sum(losses) / len(losses):.3f}, time/epoch {time.time() - start:.3f}',
                  file=sys.stderr)
            if torch.isnan(features).any() or torch.isnan(loss).any():
                print(f"NaN error! feat% {torch.isnan(features).sum() / (features.shape[0] * features.shape[1]):.3f}, "
                      f"loss% {torch.isnan(loss).sum():.3f}", file=sys.stderr)
                exit(3)

    return losses


class ContrastiveEmbedding(object):

    def __init__(
            self,
            model: torch.nn.Module,
            batch_size=32,
            negative_samples=5,
            n_iter=50,
            device="cuda:0",
            learning_rate=0.001,
            momentum=0.9,
            temperature=0.5,
            loss_mode="umap",
            optimizer="adam",
            anneal_lr=False,
            clip_grad=True,
            save_freq=25,
            callback=None,
            print_freq_epoch=None,
            print_freq_in_epoch=None,
    ):
        self.model: torch.nn.Module = model
        self.batch_size: int = batch_size
        self.negative_samples: int = negative_samples
        self.n_iter: int = n_iter
        self.device = device
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.temperature = temperature
        self.loss_mode: str = loss_mode
        self.optimizer = optimizer
        self.anneal_lr: bool = anneal_lr
        self.clip_grad: bool = clip_grad
        self.save_freq: int = save_freq
        self.callback = callback
        self.print_freq_epoch = print_freq_epoch
        self.print_freq_in_epoch = print_freq_in_epoch
        self.log_Z = None
        if self.loss_mode == "ncvis":
            self.log_Z = torch.nn.Parameter(torch.tensor(0.0),
                                            requires_grad=True)


    def fit(self, X: torch.utils.data.DataLoader):
        criterion = ContrastiveLoss(
            negative_samples=self.negative_samples,
            temperature=self.temperature,
            loss_mode=self.loss_mode,
        )

        params = self.model.parameters() if self.loss_mode != "ncvis" else \
            list(self.model.parameters()) + [self.log_Z]

        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=self.momentum,
                # weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(params,
                                         lr=self.learning_rate,)
        else:
            raise ValueError("Only optimizer 'adam' and 'sgd' allowed.")

        self.model.to(self.device)
        if self.loss_mode == "ncvis":
            self.log_Z.to(self.device)

        batch_losses = []
        for epoch in range(self.n_iter):
            lr = ((max(0.1, 1 - self.n_iter / (1 + epoch)) * self.learning_rate)
                  if self.anneal_lr
                  else self.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            bl = train(X,
                       self.model,
                       self.log_Z,
                       criterion,
                       optimizer,
                       epoch,
                       clip_grad=self.clip_grad,
                       print_freq=self.print_freq_in_epoch)
            batch_losses.append(bl)
            if (
                    self.save_freq is not None
                    and self.save_freq > 0
                    and epoch % self.save_freq == 0
                    and callable(self.callback)
            ):
                self.callback(epoch, self.model)

            if (
                    self.print_freq_epoch is not None and
                    epoch % self.print_freq_epoch == 0
            ):
                print(epoch, file=sys.stderr)

        self.losses = batch_losses
        self.embedding_ = None
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_


class ContrastiveLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, negative_samples=5,
                 temperature=0.07,
                 loss_mode='all',
                 base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.negative_samples = negative_samples
        self.temperature = temperature
        self.loss_mode = loss_mode
        self.base_temperature = base_temperature

    def forward(self, features, log_Z=None):
        """Compute loss for model. SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [2 * bsz, n_views, ...].
            log_Z: scalar, logarithm of the learnt normalization constant for ncvis.
        Returns:
            A loss scalar.
        """

        batch_size = features.shape[0] // 2
        b = batch_size

        # We can at most sample this many samples from the batch.
        # `b` can be lower than `self.negative_samples` in the last batch.
        negative_samples = min(self.negative_samples, 2 * (b - 1))

        origs = features[:b]

        # uniform probability for all other points in the minibatch,
        # except the point itself (excluded for gradient) and the
        # transformed sample (included explicitly after sampling).
        neigh_sample_weights = (torch.eye(b).repeat(1, 2) - 1) * -1 / (2 * b - 2)
        neg_inds = neigh_sample_weights.multinomial(negative_samples)

        # now add transformed explicitly
        neigh_inds = torch.hstack((torch.arange(b, 2*b)[:, None],
                                   neg_inds)).to(features.device)
        neighbors = features[neigh_inds]

        # `neigh_mask` indicates which samples feel attractive force
        # and which ones repel each other
        neigh_mask = torch.ones_like(neigh_inds, dtype=torch.bool)
        neigh_mask[:, 0] = False

        # compute probits
        diff = (origs[:, None] - neighbors)
        dists = (diff ** 2).sum(axis=2)

        # Cauchy affinities
        probits = torch.div(
                1,
                1 + dists
        )

        # probits *= neigh_mask

        if self.loss_mode == "umap":
            # cross entropy parametric umap loss
            loss = - (~neigh_mask * torch.log(probits.clamp(1e-4, 1))) \
                - (neigh_mask * torch.log((1 - probits).clamp(1e-4, 1)))
        elif self.loss_mode == "ncvis":
            probits = probits / torch.exp(log_Z)

            # for proper ncvis it should be negative_samples * p_noise. But for
            # uniform noise distribution we would need the size of the dataset
            # here. Also, we do not use a uniform noise distribution as we sample
            # negative samples from the batch.
            estimator = probits / (probits + negative_samples)
            loss = - (~neigh_mask * torch.log(estimator.clamp(1e-4, 1))) \
                - (neigh_mask * torch.log((1 - estimator).clamp(1e-4, 1)))
        else:
            # loss simclr
            loss = - (self.temperature / self.base_temperature) * (
                (torch.log(probits.clamp(1e-4, 1)[~neigh_mask]))
                - torch.log((neigh_mask * probits.clamp(1e-4, 1)).sum(axis=1))
            )

        return loss.mean()
