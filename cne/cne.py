import sys
import time

import torch
import numpy as np


def train(train_loader, model, criterion, optimizer, epoch, print_freq=None):
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
        loss = criterion(features)

        # update metric
        # losses.update(loss.item(), bsz)
        losses.append(loss.item())

        # SGD
        optimizer.zero_grad()
        # print(torch.isnan(features).any(), torch.isnan(loss).any(), file=sys.stderr)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 4)
        optimizer.step()

        # measure elapsed time
        end = time.time()

        # print info
        if print_freq is not None and (idx + 1) % print_freq == 0:
            print(f'Train: E{epoch}, {idx}/{len(train_loader)}\t'
                  f'grad magn {model.linear_relu_stack[-1].weight.grad.abs().sum()}, loss {sum(losses) / len(losses):.3f}, time/epoch {time.time() - start:.3f}',
                  file=sys.stderr)
            if torch.isnan(features).any() or torch.isnan(loss).any():
                print(f"NaN error! feat% {torch.isnan(features).sum() / (features.shape[0] * features.shape[1]):.3f}, "
                      f"loss% {torch.isnan(loss).sum():.3f}", file=sys.stderr)
                exit(3)

    print("first losses", losses[:5], "last losses", losses[-5:], file=sys.stderr)
    return sum(losses) / len(losses)


class ContrastiveEmbedding(object):

    def __init__(
            self,
            model,
            batch_size=32,
            # negative_samples=5,
            n_iter=750,
            device="cuda:0",
            learning_rate=0.005,
            momentum=0.9,
            temperature=0.5,
            save_freq=25,
            savedir=".",
            print_freq_epoch=None,
            print_freq_in_epoch=None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.device = device
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.temperature = temperature
        self.save_freq = save_freq
        self.savedir = savedir
        self.print_freq_epoch = print_freq_epoch
        self.print_freq_in_epoch = print_freq_in_epoch

    def fit(self, X: torch.utils.data.DataLoader):
        criterion = ContrastiveLoss(negative_samples=5, temperature=self.temperature)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            # weight_decay=self.weight_decay,
        )
        self.model.to(self.device)

        for epoch in range(self.n_iter):
            lr = max(0.1, 1 - self.n_iter / (1 + epoch)) * self.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            train(X,
                  self.model,
                  criterion,
                  optimizer,
                  epoch,
                  print_freq=self.print_freq_in_epoch)

            if (self.print_freq_epoch is not None and
                epoch % self.print_freq_epoch == 0):
                print(epoch, file=sys.stderr)

        self.embedding_ = None
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_


class ContrastiveLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, negative_samples=5, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.negative_samples = negative_samples
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        # if len(features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0] // 2

        b = batch_size
        m = self.negative_samples
        origs = features[:b]
        trans = features[b:]

        # uniform probability for all other points in the minibatch,
        # except the point itself (excluded for gradient) and the
        # transformed sample (included explicitly after sampling).
        neigh_sample_weights = (torch.eye(b).repeat(1, 2) - 1) * -1 / (2 * b - 2)
        neg_inds = neigh_sample_weights.multinomial(self.negative_samples)

        # now add transformed explicitly
        neigh_inds = torch.hstack((torch.arange(b, 2*b)[:, None], neg_inds)).to(device)
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

        # loss simclr
        # loss = - (self.temperature / self.base_temperature) * (
        #     torch.log(probits[~neigh_mask]) - torch.log(probits[neigh_mask].sum(axis=0, keepdim=True))
        # )

        # cross entropy parametric umap loss
        loss = - (~neigh_mask * torch.log(probits.clamp(1e-4, 1))) \
            - (neigh_mask * torch.log((1 - probits).clamp(1e-4, 1)))

        return loss.mean()
