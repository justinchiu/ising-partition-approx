import numpy as np
import torch

""" Utility functions for Ising models:
    log p(x) = -x.T @ W @ x - log Z
    log Z = logsumexp_x -x.T @ W @ x

    with x in {0,1}^n to make VI a little easier.
"""

def enumerate_support(dim):
    """ Enumerate the support of the Ising model
    """
    return torch.tensor(np.unpackbits(
        np.arange(2 ** dim, dtype=np.uint8)[:,None],
        axis=-1,
    )[:,-dim:], dtype=torch.float32)

def log_potential(W, x):
    """ Compute the log numerator of the model
    """
    return -torch.einsum("ij,bi,bj->b", W, x, x)

def log_partition(W):
    """ Compute the log partition function, log Z, via brute force.
    """
    dim = W.shape[0]
    support = enumerate_support(dim)
    log_Z = log_potential(W, support).logsumexp(0)
    return log_Z
