import torch
import numpy as np
import my_cuda_util


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
            # batch_size=2000000000,  # GPU l2
            # batch_size=10000000,  # CPU m3
            batch_size=20000000,  # GPU m3
            # batch_size=2000000000,  # GPU cuda m3
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
        self.my_cuda = kernel_cuda
        if kernel == 'l2':
            self.dist = self.cdist
        elif kernel.startswith('m'):
            self.dist = self.mdist
            self.p = int(kernel[1:])
        else:
            raise NotImplementedError

    def cdist(self, x, y, cuda=False):
        batch_size = x.shape[0] * y.shape[0] * x.shape[1] // (self.batch_size + 1) + 1
        if not cuda:
            z = [torch.cdist(x, by) for by in y.chunk(batch_size)]
            return torch.cat(z, dim=-1)
        else:
            raise NotImplementedError

    def mdist(self, x, y, cuda=False):
        batch_size = x.shape[0] * y.shape[0] * x.shape[1] // (self.batch_size + 1) + 1
        if not cuda:
            if self.device == 'cuda':
                z = [((x.unsqueeze(1) - by.unsqueeze(0)).abs() ** self.p).sum(-1) for by in y.chunk(batch_size)]
                z = torch.cat(z, dim=-1)
            else:
                z = ((x.unsqueeze(1) - y.unsqueeze(0)).abs() ** self.p).sum(-1)
        else:
            z = my_cuda_util.mdist(x, y, self.p)
            torch.cuda.synchronize()
        return z ** (1 / self.p)

    def _initial(self, X):
        select = torch.randint(0, self.n_samples, (1,), device=self.device)
        centers = torch.zeros((self.n_clusters, self.n_features), device=self.device)
        dist = torch.zeros((self.n_clusters, self.n_samples), device=self.device)

        for i in range(self.n_clusters):
            centers[i] = X.index_select(0, select)
            dist[i] = self.dist(X.index_select(0, select), X, self.my_cuda)
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
            dist = self.dist(X, centers, self.my_cuda)
            labels = torch.argmin(dist, dim=-1).view(-1)
            select = torch.sparse_coo_tensor(torch.stack([labels, x]), ones, (self.n_clusters, self.n_samples))
            weights = torch.sparse.sum(select, dim=1).to_dense()
            new_centers = torch.sparse.mm(select, X)[weights > 0] / weights[weights > 0].unsqueeze(1)
            weights = weights[weights > 0]
            self.n_clusters = len(weights)

            if centers.shape[0] == new_centers.shape[0]:
                if ((new_centers - centers).norm(dim=1) < self.tol).sum() == self.n_clusters:
                    break
            centers = new_centers.detach().clone()

        return labels.cpu().numpy(), select, weights
