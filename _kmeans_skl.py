import enum
# from _kmeans import KMeans
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.cluster import KMeans


# x = np.random.randn(1000000, 10)
# cls = KMeans(100, device='cuda')
# labels, _, _ = cls.fit_predict(x)

# print(labels)
def cifar_test(cls):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform = transform_test)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform = transform_test)

    # print(len(testset))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=50000, shuffle=True, num_workers=0)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=10000, shuffle=True, num_workers=0)

    for feature, label in trainloader:
        feature = feature.numpy().reshape(50000, -1)

        cls.fit(feature)

        y_kmeans = cls.predict(feature)


    valid_acc(testloader)


def mn_test(cls):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4465), (0.2010)),
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform = transform_test)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform = transform_test)

    # print(len(testset))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=60000, shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=60000, shuffle=True, num_workers=0)

    for feature, label in trainloader:
        feature = feature.numpy().reshape(60000,-1)
        cls.fit(feature)
        y_kmeans = cls.predict(feature)
    
    valid_acc(testloader)


def valid_acc(testloader):
    for feature, label in testloader:
        feature = feature.numpy().reshape(10000,-1)
        test_kmeans = cls.predict(feature)

        # scores = adjusted_mutual_info_score(label, test_kmeans)
        scores = adjusted_rand_score(label, test_kmeans)
        print("Accuracy: %.2f%%" % (scores*100))


# def valid_acc(testloader):
#     label_test = []
#     for i in range(10):
#         tmp = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
#         label_test.append(tmp)
    
#     _correct = 0
#     _tot = 0
#     for feature, label in testloader:
#         feature = feature.numpy().reshape(10000,-1)
#         test_kmeans = cls.predict(feature)
#         for i in range(len(label)):
#             label_test[test_kmeans[i]][label[i].item()] += 1

#         label_cluster = []
#         for i in range(10):
#             _max_num = 0
#             _max_idx = 0
#             for j in range(10):
#                 _tot += label_test[i][j]
#                 if _max_num < label_test[i][j]:
#                     _max_num, _max_idx = label_test[i][j], j
            
#             _correct += _max_num
#             label_cluster.append(_max_idx)
#         print(label_cluster)
#     print("Accuracy: %.2f%%" % (_correct/_tot * 100))

if __name__ == '__main__':
    cls = KMeans(init="k-means++", n_clusters=10, n_init=4, random_state=0)
    # cifar_test(cls)
    mn_test(cls)



    


