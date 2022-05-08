import torch
import numpy as np


class GaussianMixture:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_predict(self, x, select, weights, delta=1.e-3, eps=1.e-6, n_iter=100, top=10):  # (n, d,)
        self.n_components = len(weights)
        ind = select.coalesce().indices()
        self.batch = x.shape[0] * self.n_components // 4000001 + 1
        self.log2pi = -.5 * x.shape[1] * np.log(2. * np.pi)
        self.mu = torch.sparse.mm(select, x) / weights.unsqueeze(1)
        self.var = torch.stack([(x.index_select(0, ind[1][ind[0] == i]) - self.mu[i]).pow(2).mean(0)
                                for i in range(self.mu.shape[0])]) + eps
        self.pi = (weights / x.shape[0]).unsqueeze(0) + eps

        probs, self.log_likelihood = self._e_step(x)
        resp = torch.nn.functional.softmax(probs, dim=1)

        for i in range(n_iter):
            log_likelihood_old = self.log_likelihood
            resp_old = resp

            # m_step & update
            self._m_step(x, resp)
            torch.cuda.synchronize()

            # e_step
            probs, self.log_likelihood = self._e_step(x)
            resp = torch.nn.functional.softmax(probs, dim=1)

            # check convergence
            if (self.log_likelihood - log_likelihood_old < delta or
                    ((resp_old - resp).norm(dim=0) < delta).sum() == self.n_components):
                break

        top = min(top, self.n_components)
        probs = torch.nn.functional.softmax(probs, dim=1)
        topk = probs.topk(top, dim=1)
        probs = topk.values.view(-1)
        labels = topk.indices.view(-1)[probs > .1]
        x_ind = torch.arange(x.shape[0], device=device).repeat_interleave(top)[probs > .1]
        probs = probs[probs > .1]
        ind = torch.stack([labels, x_ind])
        select = torch.sparse_coo_tensor(ind, probs, (self.n_components, x.shape[0]))
        weights = torch.sparse.sum(select, dim=1).to_dense()

        return select, weights

    def _m_step(self, x, resp, eps=1.e-6):
        self.pi = torch.sum(resp, dim=0, keepdim=True) + eps
        self.mu = torch.mm(resp.T, x) / self.pi.T
        self.var = torch.stack([
            (r * (bx - self.mu.unsqueeze(0)).pow(2)).sum(0) for r, bx in
            zip(resp.unsqueeze(2).chunk(self.batch), x.unsqueeze(1).chunk(self.batch))
        ]).sum(0) / self.pi.T + eps
        self.pi /= x.shape[0]

    def _e_step(self, x):
        var_log_sum = -.5 * self.var.log().sum(dim=1).unsqueeze(0)
        log_prob = -.5 * torch.cat([((bx.unsqueeze(1) - self.mu.unsqueeze(0)).pow(2) / self.var.unsqueeze(0)).sum(dim=2)
                                    for bx in x.chunk(self.batch)], dim=0) + var_log_sum + self.log2pi + self.pi.log()
        log_likelihood = torch.logsumexp(log_prob, dim=1).mean()  # (n, 1,)
        return log_prob, log_likelihood
