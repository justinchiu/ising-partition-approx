import numpy as np
import torch
import pyro
import math

from ising.model import enumerate_support, log_partition
from ising.vi import InferenceNetwork, fit

def test_brute_force():
    dim = 4
    xs = enumerate_support(dim)
    W = torch.ones(dim, dim)
    log_Z = log_partition(W)
    assert math.fabs(log_Z.item() - 0.9485276) < 1e-5

def test_vi_easy():
    dim = 4
    W = torch.ones(dim, dim)
    log_Z = log_partition(W)

    model = InferenceNetwork(W)
    fit(model, lr=1e-2)

    print("True", log_Z.item())
    print("Lowerbound", -model.upperbound().item())

    assert log_Z.item() > -model.upperbound().item()
    assert log_Z.item() + model.upperbound().item() < 1e-3
