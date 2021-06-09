import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


def cluster_weights(model, n_clusters=2):
    for layer in model.decoder.parameters():
        data = layer.detach().numpy()
        data = data.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters).fit(data)

        for cluster in range(n_clusters):
            t1 = np.float32(kmeans.labels_ == cluster).reshape(-1, 1)
            t2 = 1-t1

            data = data*t2
            t1 = t1 * kmeans.cluster_centers_[cluster]
            data += t1

        layer.data = nn.Parameter(torch.from_numpy(data.reshape(layer.shape)))
        bp = 0

