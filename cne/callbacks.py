from openTSNE.callbacks import Callback
import torch
import numpy as np
try:
    from vis_utils.utils import compute_normalization, expected_loss_keops, \
        NCE_loss_keops, KL_divergence
    vis_utils_available = True
except:
    vis_utils_available = False

class Logger(Callback):
    def __init__(self,
                 log_embds=False,
                 log_losses=False,
                 loss_type=None,
                 graph=None,
                 log_norms=False,
                 log_kl=False):

        self.log_embds = log_embds
        if self.log_embds:
            self.embds = []

        self.graph = graph.tocoo() if graph is not None else graph

        self.log_losses = log_losses
        self.loss_type = loss_type
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

    def __call__(self, epoch, model, negative_samples, loss_mode, log_Z=None, noise_in_estimator=None):
        if self.log_embds or self.log_norm or self.log_losses or self.log_kl:
            assert isinstance(model, torch.nn.modules.sparse.Embedding), \
            f"To log the model must be of type torch.nn.modules.sparse.Embedding but is of type {type(model)}"
            embd = model.weight.detach().cpu().numpy()

        if self.log_embds:
            self.embds.append(embd)

        if self.log_losses:
            if loss_mode == "UMAP":
                self.losses.append(expected_loss_keops(high_sim=self.graph,
                                                       embedding=embd,
                                                       a=1.0,
                                                       b=1.0,
                                                       negative_sample_rate=negative_samples,
                                                       push_tail=True))
            elif loss_mode == "ncvis":
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
            elif loss_mode == "neg_sample":
                # turn noise_in_ratio back into Z via
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
            self.norms.append(compute_normalization(embd,
                                                    sim_func="cauchy",
                                                    no_diag=True,
                                                    a=1.0,
                                                    b=1.0,
                                                    eps=float(np.finfo(float).eps)))



