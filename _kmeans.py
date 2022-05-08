import torch
import numpy as np


class KMeans:
    def __init__(
            self,
            n_clusters,
            kernel='l2',
            kernel_cuda=False,
            random_state=42,
            device=None,
            max_iter=300,
            tol=1e-4,
            batch_size=200000000
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.batch_size = batch_size
        self.labels_ = None
        self.select_ = None
        self.X_dist = None

    def _initial(self, X):
        select = torch.randint(0, self.n_samples, (1,), device=self.device)
        centers = torch.zeros((self.n_clusters, self.n_features), device=self.device)
        dist = torch.zeros((self.n_clusters, self.n_samples), device=self.device)

        for i in range(self.n_clusters):
            centers[i] = X.index_select(0, select)
            dist[i] = torch.cdist(X.index_select(0, select), X)
            if i == 0:
                minimum = dist[0]
            else:
                minimum = torch.min(dist[i], minimum)
            select = torch.argmax(minimum)
        return centers

    def fit_predict(self, X, gamma=None):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.batch = self.n_samples * self.n_clusters // (self.batch_size + 1) + 1

        labels, select, weights = None, None, None

        if type(X) != torch.Tensor:
            X = torch.from_numpy(X).float().to(self.device)
        centers = self._initial(X)
        x = torch.arange(self.n_samples, device=self.device)
        ones = torch.ones_like(x).float()
        for _iter in range(self.max_iter):
            labels = torch.cat([torch.argmin(torch.cdist(bx, centers), dim=1).view(-1) for bx in X.chunk(self.batch)])
            select = torch.sparse_coo_tensor(torch.stack([labels, x]), ones, (self.n_clusters, self.n_samples))
            weights = torch.sparse.sum(select, dim=1).to_dense()
            new_centers = torch.sparse.mm(select, X)[weights > 0] / weights[weights > 0].unsqueeze(1)
            weights = weights[weights > 0]
            self.n_clusters = len(weights)

            if centers.shape[0] == new_centers.shape[0]:
                if ((new_centers - centers).norm(dim=1) < self.tol).sum() == self.n_clusters:
                    break
            centers = new_centers.detach().clone()

        if gamma is not None:
            m = torch.nn.Softmax(dim=1)
            dist = torch.cat([m(-gamma * torch.cdist(bx, centers).pow(2)) for bx in X.chunk(self.batch)], dim=0)
            sort = torch.sort(dist, dim=1)
            probs = sort.values.view(-1)
            labels = sort.indices.view(-1)[probs > .95 / self.n_clusters]
            x_ind = torch.arange(x.shape[0], device=self.device).repeat_interleave(
                self.n_clusters)[probs > .95 / self.n_clusters]
            probs = probs[probs > .95 / self.n_clusters]
            ind = torch.stack([labels, x_ind])
            select = torch.sparse_coo_tensor(ind, probs, (self.n_clusters, x.shape[0]))
            try:
                weights = torch.sparse.sum(select, dim=1).to_dense()
            except:
                print(select)

        return labels.cpu().numpy(), select, weights
