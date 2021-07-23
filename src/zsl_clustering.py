import torch
import torch.nn as nn
import sys
import numpy as np
from sklearn.cluster import KMeans

import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import AgglomerativeClustering


AVAILABLE_CLUSTERING = ['KMeans', 'Birch', 'AgglomerativeClustering', 'SpectralClustering']

# def cluster_weights(model, n_clusters=2):
#
#     print("#-------------------------------------------------------------------------------")
#     print(f'Clustering with KMeans...')
#     print("#-------------------------------------------------------------------------------")
#
#     for layer in model.decoder.parameters():
#         data = layer.detach().numpy()
#         data = data.reshape(-1, 1)
#         kmeans = KMeans(n_clusters=n_clusters).fit(data)
#
#         for cluster in range(n_clusters):
#             t1 = np.float32(kmeans.labels_ == cluster).reshape(-1, 1)
#             t2 = 1-t1
#
#             data = data*t2
#             t1 = t1 * kmeans.cluster_centers_[cluster]
#             data += t1
#
#         layer.data = nn.Parameter(torch.from_numpy(data.reshape(layer.shape)))
#         bp = 0


def cluster_weights(model, n_clusters=100, cluster_name='KMeans'):

    if cluster_name not in AVAILABLE_CLUSTERING:
        print(f'\n[ERROR]: {cluster_name} is not a supported algorithm.')
        print('[ERROR]: Supported algorithms: ')

        print('\t\t[')
        for name in AVAILABLE_CLUSTERING:
            print(f'\t\t{name},')
        print('\t\t]\n')

        sys.exit()

    print("#-------------------------------------------------------------------------------")
    print(f'Clustering with {cluster_name}...')
    print("#-------------------------------------------------------------------------------")

    for layer in model.decoder.parameters():
        data = layer.detach().numpy()
        data = data.reshape(-1, 1)

        if cluster_name == 'KMeans':
            kmeans = KMeans(n_clusters=n_clusters).fit(data)

            for cluster in range(n_clusters):
                t1 = np.float32(kmeans.labels_ == cluster).reshape(-1, 1)
                t2 = 1-t1

                data = data*t2
                t1 = t1 * kmeans.cluster_centers_[cluster]
                data += t1

            layer.data = nn.Parameter(torch.from_numpy(data.reshape(layer.shape)))
            bp = 0
        if cluster_name == 'Birch':
            birch = Birch(threshold=0.0001, n_clusters=None).fit(data)

            for cluster in range(len(birch.subcluster_centers_)):
                t1 = np.float32(birch.labels_ == cluster).reshape(-1, 1)
                t2 = 1 - t1

                data = data * t2
                t1 = t1 * birch.subcluster_centers_[cluster]
                data += t1

            layer.data = nn.Parameter(torch.from_numpy(data.reshape(layer.shape)))
            bp = 0
        if cluster_name == 'AgglomerativeClustering':
            optics = AgglomerativeClustering(n_clusters=n_clusters).fit(data)

            for cluster in range(len(np.unique(optics.labels_))):
                t1 = np.float32(optics.labels_ == cluster).reshape(-1, 1)
                t2 = 1 - t1

                cluster_center = np.mean(data[t1 != 0])

                data = data * t2
                t1 = t1 * cluster_center
                data += t1

            layer.data = nn.Parameter(torch.from_numpy(data.reshape(layer.shape)))
            bp = 0
        if cluster_name == 'SpectralClustering':
            optics = SpectralClustering(n_clusters=n_clusters).fit(data)

            for cluster in range(len(np.unique(optics.labels_))):
                t1 = np.float32(optics.labels_ == cluster).reshape(-1, 1)
                t2 = 1 - t1

                cluster_center = np.mean(data[t1 != 0])

                data = data * t2
                t1 = t1 * cluster_center
                data += t1

            layer.data = nn.Parameter(torch.from_numpy(data.reshape(layer.shape)))
            bp = 0


