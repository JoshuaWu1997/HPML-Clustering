from _kmeans import KMeans
import numpy as np

x = np.random.randn(1000000, 10)
cls = KMeans(100, device='cuda')
labels, _, _ = cls.fit_predict(x)

print(labels)
