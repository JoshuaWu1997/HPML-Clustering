import enum
from _kmeans import KMeans
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from sklearn.cluster import KMeans


# x = np.random.randn(1000000, 10)
# cls = KMeans(100, device='cuda')
# labels, _, _ = cls.fit_predict(x)

# print(labels)

if __name__ == '__main__':
    cls = KMeans(init="k-means++", n_clusters=10, n_init=4, random_state=0)

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

        y_kmeans = cls.predict(feature)


