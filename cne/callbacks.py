from openTSNE.callbacks import Callback
import torch
import numpy as np
try:
    from vis_utils.utils import compute_normalization
    vis_utils_available = True
except:
    vis_utils_available = False

class Logger(Callback):
    def __init__(self,
                 log_embds=False,
                 log_losses=False,
                 loss_type=None,
                 log_norms=False):

        self.log_embds = log_embds
        if self.log_embds:
            self.embds = []

        # to log losses we would need the graph, which we do not have atm...
        #self.log_losses = log_losses
        #self.loss_type = loss_type
        #if self.log_losses:
        #    assert vis_utils, f"Need vis_utils package to log losses."
        #    self.losses = []

        self.log_norm = log_norms
        if self.log_norm:
            assert vis_utils_available, "Need vis_utils package to log normalization constant."
            self.norms = []

    def __call__(self, epoch, model):
        if self.log_embds:
            assert isinstance(model, torch.nn.modules.sparse.Embedding), \
            f"To log embeddings model must be of type torch.nn.modules.sparse.Embedding but is of type {type(model)}"
            self.embds.append(model.weight.detach().cpu().numpy())

        if self.log_norm:
            self.norms.append(compute_normalization(model.weight.detach().cpu().numpy(),
                                                    sim_func="cauchy",
                                                    no_diag=True,
                                                    a=1.0,
                                                    b=1.0,
                                                    eps=float(np.finfo(float).eps)))



