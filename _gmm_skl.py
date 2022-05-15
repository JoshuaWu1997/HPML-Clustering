import enum
# from _kmeans import KMeans
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from sklearn.mixture import GaussianMixture


# x = np.random.randn(1000000, 10)
# cls = KMeans(100, device='cuda')
# labels, _, _ = cls.fit_predict(x)

# print(labels)

def cifar_test():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform = transform_test)

    # print(len(trainset))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=50000, shuffle=True, num_workers=0)

    for feature, label in trainloader:
        feature = feature.numpy().reshape(50000,-1)

        cls.fit(feature)
        print("finish")

        y_kmeans = cls.predict(feature)

def mn_test():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4465), (0.2010)),
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform = transform_test)

    # print(len(trainset))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=60000, shuffle=True, num_workers=0)

    for feature, label in trainloader:
        feature = feature.numpy().reshape(60000,-1)

        cls.fit(feature)
        print("finish")

        y_kmeans = cls.predict(feature)

if __name__ == '__main__':
    cls = GaussianMixture(n_components=10, covariance_type='full', random_state=0)
    cifar_test()
    # mn_test()




