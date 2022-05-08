from options import params
from _kmeans import KMeans
from _gaussian import GaussianMixture
import os
import torch
import torchvision
import torchvision.transforms as transforms
import time
import pandas as pd

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def CIFAR_test(cfg):
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=50000, shuffle=True, num_workers=0)
    for feature, label in trainloader:
        feature = feature.view(50000, -1).to(cfg.device)
        if cfg.alg == 'kmeans++':
            cls = KMeans(10, kernel=cfg.kernel, kernel_cuda=cfg.kernel_cuda, device=cfg.device)
        else:
            cls = GaussianMixture(10, device=cfg.device)
        start = time.time_ns()
        _ = cls.fit_predict(feature)
        end = time.time_ns()
        return (end - start) / 1000000000


def Synthetic_test(cfg):
    feature = torch.randn(1000000, 1000).to(cfg.device)
    if cfg.alg == 'kmeans++':
        cls = KMeans(1000, kernel=cfg.kernel, kernel_cuda=cfg.kernel_cuda, device=cfg.device)
    else:
        cls = GaussianMixture(1000, device=cfg.device)
    start = time.time_ns()
    _ = cls.fit_predict(feature)
    end = time.time_ns()
    return (end - start) / 1000000000


if __name__ == '__main__':
    metrics = dict()
    os.makedirs('outputs', exist_ok=True)
    for i in range(10):
        if params.dataset == 'CIFAR':
            use_time = CIFAR_test(params)
        elif params.dataset == 'Synthetic':
            use_time = CIFAR_test(params)
        metrics[i] = {'time': use_time}
    pd.DataFrame(metrics).to_csv(f'outputs/{params.dataset}-{params.alg}-'
                                 f'{params.kernel}-{params.kernel_cuda}'
                                 f'-{params.device}-{params.gpu_type}')
