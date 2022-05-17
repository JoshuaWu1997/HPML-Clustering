from options import params
from _kmeans import KMeans
from _gaussian import GaussianMixture
import os
import torch
import torchvision
import torchvision.transforms as transforms
import time
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def MNIST_test(cfg):
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=60000, shuffle=True, num_workers=0)
    for feature, label in trainloader:
        feature = feature.view(60000, -1).to(cfg.device)
        if cfg.alg == 'kmeans++':
            cls = KMeans(10, kernel=cfg.kernel, kernel_cuda=cfg.kernel_cuda, device=cfg.device)
        else:
            cls = GaussianMixture(10, device=cfg.device)
        start = time.time_ns()
        predict, _, _ = cls.fit_predict(feature)
        end = time.time_ns()
        metric = adjusted_mutual_info_score(predict, label.numpy())
        print('ACC:', metric)
        return (end - start) / 1000000000

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
        predict, _, _ = cls.fit_predict(feature)
        end = time.time_ns()
        metric = adjusted_mutual_info_score(predict, label.numpy())
        print('ACC:', metric)
        return (end - start) / 1000000000


def CIFAR100_test(cfg):
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_test)
    print(len(trainset))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=50000, shuffle=True, num_workers=0)
    for feature, label in trainloader:
        feature = feature.view(50000, -1).to(cfg.device)
        if cfg.alg == 'kmeans++':
            cls = KMeans(100, kernel=cfg.kernel, kernel_cuda=cfg.kernel_cuda, device=cfg.device)
        else:
            cls = GaussianMixture(100, device=cfg.device)
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
        elif params.dataset == 'CIFAR100':
            use_time = CIFAR100_test(params)
        elif params.dataset == 'MNIST':
            use_time = MNIST_test(params)
        metrics[i] = {'time': use_time}
    pd.DataFrame(metrics).to_csv(f'outputs/{params.dataset}-{params.alg}-'
                                 f'{params.kernel}-{params.kernel_cuda}'
                                 f'-{params.device}-{params.gpu_type}')
