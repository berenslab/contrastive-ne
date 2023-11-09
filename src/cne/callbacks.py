import torch
import numpy as np
try:
    from vis_utils.utils import compute_normalization, expected_loss_keops, \
        NCE_loss_keops, KL_divergence
    vis_utils_available = True
except:
    vis_utils_available = False

class Logger():
    """
    Class for logging various quantities of interest during a contrastive neighbor embedding optimization.
    """
    def __init__(self,
                 log_embds=False,
                 log_losses=False,
                 graph=None,
                 log_norms=False,
                 log_kl=False,
                 n=None):
        """
        :param log_embds: bool If true, log intermediate embeddings.
        :param log_losses: bool If true, log the (expected) loss of the model.
        :param graph: sparse matrix Holds the similarity graph.
        :param log_norms: bool If true, log the intermediate values of the parition function.
        :param log_kl: bool If true, log the intermediate values of the KL divergence.
        :param n: int Dataset size
        """

        self.log_embds = log_embds
        if self.log_embds:
            self.embds = []
            self.Zs = []
        else:
            self.embds = None

        self.graph = graph.tocoo() if graph is not None else graph

        self.log_losses = log_losses
        if self.log_losses:
            assert vis_utils_available, f"Need vis_utils package to log losses."
            self.losses = []

        self.log_kl = log_kl
        if self.log_kl:
            assert vis_utils_available
            assert self.graph is not None
            self.kls = []

        self.log_norm = log_norms
        if self.log_norm:
            assert vis_utils_available, "Need vis_utils package to log normalization constant."
            self.norms = []

        if n is not None:
            self.ds = torch.utils.data.TensorDataset(torch.arange(n))
            self.dl = torch.utils.data.DataLoader(self.ds,
                                                  batch_size=256,
                                                  shuffle=False)

    def __call__(self, epoch, model, negative_samples, loss_mode, log_Z=None, noise_in_estimator=None):
        # read out the embeddings from the model if anything shall be logged
        if isinstance(model, torch.nn.modules.sparse.Embedding):
            # non-parametric case, just get all embeddings from embedding layer
            embd = model.weight.detach().cpu().numpy()
        else:
            # parametric case, model is Embedding layer + FCNetwork
            # Just feed indices from self.dl
            device = model[0].weight.device
            embd = np.vstack([model(batch[0].to(device))
                             .detach().cpu().numpy()
                              for batch in self.dl])
        if log_Z is not None and self.log_embds:
            self.Zs.append(torch.exp(log_Z).detach().cpu().numpy())

        if self.log_embds:
            self.embds.append(embd)
        else:
            self.embds = [embd]

        if self.log_losses:
            if loss_mode == "umap":
                self.losses.append(expected_loss_keops(high_sim=self.graph,
                                                       embedding=embd,
                                                       a=1.0,
                                                       b=1.0,
                                                       negative_sample_rate=negative_samples,
                                                       push_tail=True))
            elif loss_mode == "nce":
                # transform Z since the implemented criterion does not consider the noise distribution explicitly
                Z_prime = np.exp(log_Z.detach().cpu().numpy())
                Z_prime = Z_prime * len(embd)**2
                self.losses.append(NCE_loss_keops(high_sim=self.graph,
                                                  embedding=embd,
                                                  m=negative_samples,
                                                  Z=Z_prime,
                                                  a=1.0,
                                                  b=1.0,
                                                  noise_log_arg=True,
                                                  eps=1e-4))
            elif loss_mode == "neg":
                # turn noise_in_estimator back into Z via
                # Z * m * p_n = 1 <--> Z = 1 / (m * p_n)
                Z = (negative_samples / len(embd)**2)**-1
                self.losses.append(NCE_loss_keops(high_sim=self.graph,
                                                  embedding=embd,
                                                  m=negative_samples,
                                                  Z=Z,
                                                  a=1.0,
                                                  b=1.0,
                                                  noise_log_arg=True,
                                                  eps=1e-4))
            else:
                raise NotImplementedError("No expected loss function is implemented for InfoNCE")

        if self.log_kl:
            self.kls.append(KL_divergence(self.graph,
                                          embedding=embd,
                                          a=1.0,
                                          b=1.0,
                                          sim_func="cauchy",
                                          eps = float(np.finfo(float).eps),
                                          norm_over_pos=False)
                            )
        if self.log_norm:
            norm = compute_normalization(embd,
                                         sim_func="cauchy",
                                         no_diag=True,
                                         a=1.0,
                                         b=1.0,
                                         eps=float(np.finfo(float).eps))
            self.norms.append(norm.detach().cpu().numpy())



