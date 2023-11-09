import cne
import numpy as np

# get random data
x1 = np.random.randn(1000, 10)
x2 = np.random.randn(1000, 10) + np.ones((1000, 10))
x = np.concatenate([x1, x2])

# non-parametric Neg-t-SNE
embedder_neg = cne.CNE(loss_mode="neg",
                       k=15,
                       optimizer="sgd",
                       parametric=False,
                       print_freq_epoch=10)
embd_neg = embedder_neg.fit_transform(x)
print("Test OK!")