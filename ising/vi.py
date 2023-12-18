""" Simple varational inference for a sort of Ising model.
phi(x) = -x.T @ W @ x
log p(x) = phi(x) - log Z
log q(x) = sum_i log q(x[i]), q(x[i]) = Bernoulli(mu[i])

To find the best q, standard VI: minimize
KL[q || p]
= E_q[log q(x) - log p(x)]
= E_q[log q(x)] - E_q[phi(x) - log Z]

This yields a lower bound on the log partition function
log Z
= -H[q] - E_q[phi(x)] - KL[q || p]
>= -H[q] - E_q[phi(x)],
achieved iff q == p (which won't happen due to misspecification)
"""
import torch

class InferenceNetwork(torch.nn.Module):
    """ Fully factored inference network for Ising model.
        Parameterizes vector of Bernoulli means, mu, independently.
    """
    def __init__(self, W):
        super().__init__()
        dim = W.shape[0]
        self.W = W
        self.means = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def entropy(self):
        """ Sum of entropies of iid Bernoullis
            H[q] = \sum_i H[q_i]
            H[q_i] = mu_i \log mu_i + (1-mu_i) \log (1-mu_i)
        """
        mu = self.means.sigmoid()
        complement = 1 - mu
        return -(
            (mu * mu.log()).sum()
            + (complement * complement.log())
        ).sum()

    def expected_score(self):
        """ E_q[phi(x)]
            = E_q[-x.T @ W @ x]
            = -\sum_{i,j} E_{xi, xj ~ q}[x[i] * x[j] * W[i,j]]
            = -\sum_{i,j} E_{xi}[E_{xi | xj}[x[i] * x[j] * W[i,j]]]
            = -(\sum_{i,j} mu[i] * mu[j] * W[i,j]
                - \sum_{i=j} mu[i]^2 * W[i,i]
                + \sum_{i=j} mu[i] * W[i,i]
               )
            = -(\sum_{i,j} mu[i] * mu[j] * W[i,j]
                + \sum_{i=j} mu[i](1-mu[i]) * W[i,i]
               )
            = -\sum_{i,j} mu[i] * mu[j] * W[i,j] - 2*\sum_i mu[i] * (1-mu[i]) * W[i,i]
        """
        mu = self.means.sigmoid()
        print(mu)
        quadratic = torch.einsum("i,j,ij->", mu, mu, self.W)
        bias = torch.einsum("i,i,i->", mu, 1-mu, self.W.diag())
        return -quadratic - 2*bias

    def upperbound(self):
        return self.entropy() + self.expected_score()

def fit(model, num_steps=100, lr=1e-2):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = model.upperbound()
        print(model.entropy().item(), model.expected_score().item())
        loss.backward()
        print(loss.item())
        optimizer.step()

if __name__ == "__main__":
    from ising.model import log_partition
    dim = 4
    #W = torch.ones(dim, dim)
    W = torch.randn(dim, dim)
    log_Z = log_partition(W)

    model = InferenceNetwork(W)
    fit(model, lr=1e-2, num_steps=200)

    print("True", log_Z.item())
    print("Lowerbound", -model.upperbound().item())
