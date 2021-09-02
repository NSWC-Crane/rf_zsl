import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import datetime
from numpy import diff
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

device = "cpu"


def init_weights(model):
    # use something like this to manually set the weights.  use the no_grad() to prevent tracking of gradient changes
    with torch.no_grad():
        for param in model.decoder.parameters():
            # rnd_range = 1/128
            mr = np.random.default_rng(10)
            param_size = param.size()
            param.data = nn.Parameter(
                torch.from_numpy(2 * (mr.uniform(0, 1.0, param.size()) > 0.5).astype(np.float32) - 1)).to(device)

            # make a deep copy of the weights to make sure they don't change
            ew1 = copy.deepcopy(param)


def set_avg_weights(model, mode=2):
    with torch.no_grad():
        for param in model.decoder.parameters():
            data = param.data

            if mode == 0:
                r, c = param.size()

                t1 = nn.Parameter(torch.from_numpy(np.float32(data > 0)))
                p_avg = torch.sum(t1 * data, 0) / torch.sum(t1, 0)
                p_avg = p_avg.view(-1, 1).repeat(r, 1).view(r, c)
                t1 = t1 * p_avg

                t2 = nn.Parameter(torch.from_numpy(np.float32(data < 0)))
                n_avg = torch.sum(t2 * data, 0) / torch.sum(t2, 0)
                n_avg = n_avg.view(-1, 1).repeat(r, 1).view(r, c)
                t2 = t2 * n_avg

                param.data = (t1 + t2)

                bp = 0
            elif mode == 1:
                r, c = param.size()

                t1 = nn.Parameter(torch.from_numpy(np.float32(data > 0)))
                p_avg = torch.sum(t1 * data, 1) / torch.sum(t1, 1)
                p_avg = p_avg.view(-1, 1).repeat(1, c).view(r, c)
                t1 = t1 * p_avg

                t2 = nn.Parameter(torch.from_numpy(np.float32(data < 0)))
                n_avg = torch.sum(t2 * data, 1) / torch.sum(t2, 1)
                n_avg = n_avg.view(-1, 1).repeat(1, c).view(r, c)
                t2 = t2 * n_avg

                param.data = (t1 + t2)

                bp = 0
            elif mode == 2:
                t1 = nn.Parameter(torch.from_numpy(np.float32(data > 0)))
                p_avg = torch.sum(t1 * data) / torch.sum(t1)
                t1 = t1 * p_avg

                t2 = nn.Parameter(torch.from_numpy(np.float32(data < 0)))
                n_avg = torch.sum(t2 * data) / torch.sum(t2)
                t2 = t2 * n_avg

                param.data = (t1 + t2)
            else:
                print(f'Mode = ({mode}) is an invalid mode')


def set_norm_weights(model):
    for param in model.decoder.parameters():
        data = param.data

        t1 = nn.Parameter(torch.from_numpy(np.float32(data > 0)))
        d1 = t1 * data
        t1_mean = torch.mean(d1).detach().numpy()
        t1_std = torch.std(d1).detach().numpy()
        d1 = torch.normal(mean=torch.mean(d1).detach(), std=torch.std(d1).detach() / 2,
                          size=list(torch.tensor(d1.size()).numpy()))
        d1 = d1 * t1

        t2 = nn.Parameter(torch.from_numpy(np.float32(data < 0)))
        d2 = t2 * data
        t2_mean = torch.mean(d2).detach().numpy()
        t2_std = torch.std(d2).detach().numpy()
        d2 = torch.normal(mean=torch.mean(d2).detach(), std=torch.std(d2).detach() / 2,
                          size=list(torch.tensor(d2.size()).numpy()))
        d2 = d2 * t2

        param.data = (d1 + d2)

        bp = 0


def get_dist(model):
    l = []

    for param in model.decoder.parameters():
        data = param.data

        t1 = nn.Parameter(torch.from_numpy(np.float32(data > 0)))
        t2 = nn.Parameter(torch.from_numpy(np.float32(data < 0)))

        t1_sum = torch.sum(t1)
        t2_sum = torch.sum(t2)

        l.append(t1_sum - t2_sum)

        bp = 0
    return l


def cluster_weights(model, n_clusters=2):
    for param in model.decoder.parameters():
        data = param.detach().numpy()
        data = data.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters).fit(data)

        for cluster in range(n_clusters):
            t1 = np.float32(kmeans.labels_ == cluster).reshape(-1, 1)
            t2 = 1 - t1

            data = data * t2
            t1 = t1 * kmeans.cluster_centers_[cluster]
            data += t1

        param.data = nn.Parameter(torch.from_numpy(data.reshape(param.shape)))
        bp = 0


class DevLoss(nn.Module):

    def __init__(self):
        super(DevLoss, self).__init__()

    def forward(self, model):
        p_loss = 0
        n_loss = 0

        data = model.decoder.input_layer.weight

        t1 = nn.Parameter(torch.from_numpy(np.float32(data > 0)))
        t1 = t1 * data
        p_loss += torch.std(t1)

        t2 = nn.Parameter(torch.from_numpy(np.float32(data < 0)))
        t2 = t2 * data
        n_loss += torch.std(t2)

        data = model.decoder.output_layer.weight

        t1 = nn.Parameter(torch.from_numpy(np.float32(data > 0)))
        t1 = t1 * data
        p_loss += torch.std(t1)

        t2 = nn.Parameter(torch.from_numpy(np.float32(data < 0)))
        t2 = t2 * data
        n_loss += torch.std(t2)

        total_loss = p_loss + n_loss
        return total_loss


class MeanLoss(nn.Module):

    def __init__(self, m):
        super(MeanLoss, self).__init__()
        self.m = m

    def forward(self, model):
        p_loss = 0
        n_loss = 0

        data = model.decoder.input_layer.weight

        t1 = nn.Parameter(torch.from_numpy(np.float32(data > 0)))
        t1 = t1 * data
        p_loss += torch.abs(torch.mean(t1) - (1 / self.m))

        t2 = nn.Parameter(torch.from_numpy(np.float32(data < 0)))
        t2 = t2 * data
        n_loss += torch.abs(torch.mean(t2) - (1 / self.m))

        data = model.decoder.output_layer.weight

        t1 = nn.Parameter(torch.from_numpy(np.float32(data > 0)))
        t1 = t1 * data
        p_loss += torch.abs(torch.mean(t1) - (1 / self.m))

        t2 = nn.Parameter(torch.from_numpy(np.float32(data < 0)))
        t2 = t2 * data
        n_loss += torch.abs(torch.mean(t2) - (1 / self.m))

        total_loss = p_loss + n_loss
        return total_loss


class SmallWeightLoss(nn.Module):

    def __init__(self):
        super(SmallWeightLoss, self).__init__()

    def forward(self, model):
        p_loss = 0
        n_loss = 0

        data = model.decoder.input_layer.weight

        t1 = nn.Parameter(torch.from_numpy(np.float32(data > 0)))
        t1 = t1 * data
        p_loss += 1 / torch.sum(t1) * 1000000

        t2 = nn.Parameter(torch.from_numpy(np.float32(data < 0)))
        t2 = t2 * data
        n_loss += 1 / torch.sum(t2) * 1000000

        data = model.decoder.output_layer.weight

        t1 = nn.Parameter(torch.from_numpy(np.float32(data > 0)))
        t1 = t1 * data
        p_loss += 1 / torch.sum(t1) * 1000000

        t2 = nn.Parameter(torch.from_numpy(np.float32(data < 0)))
        t2 = t2 * data
        n_loss += 1 / torch.sum(t2) * 1000000

        total_loss = p_loss + (-1 * n_loss)
        return total_loss


def get_feature_size(in_size):
    return [int(in_size / 1.2), int(in_size / 2), int(in_size / 3),
            int(in_size / 4), int(in_size / 5), int(in_size / 6)]


def get_n_cluster(fs):
    return [int(fs / 2), int(fs / 3),
            int(fs / 4), int(fs / 5), int(fs / 6)]


def get_d_init(fs):
    return [int(fs * 1.5),
            int(fs * 2), int(fs * 4)]


def count_freq(labels, max=200):
    """
    return an array of tuples, containing the frequency a particular value appears
    input: list, label for each weight
    output: list
    """

    unique, count = np.unique(labels, return_counts=True)
    return unique, count


def sort_labels(labels_arr, map_arr):
    """Rearrange labels based on the map_arr"""

    sorted_labels = np.zeros(labels_arr.shape) - 1

    # loop from 0, 1, 2... n_clusters
    # replace each labels with its mapped index
    for label, mapped_idx in enumerate(map_arr):
        idx = np.where(labels_arr == mapped_idx)[0]
        sorted_labels[idx] = label

    return sorted_labels


def get_ranges(discontinuities, last_idx=1023):

    if discontinuities.size == 0:
        return [(0, last_idx)]

    r = [(0, discontinuities[0])]

    for i in range(1, discontinuities.size):
        r.append((discontinuities[i - 1] + 1, discontinuities[i]))
    r.append((discontinuities[-1] + 1, last_idx))

    return r


def search_range(label, ranges):
    for idx, trange in enumerate(ranges):
        if trange[0] <= label <= trange[1]:
            return idx

    return None


### GRAPHS ###


def plot_data(y, x=None, title='', xlabel='', ylabel='', file_name='', save=False):

    y = np.reshape(y, (-1,))

    plt.figure()
    if x is not None:
        plt.plot(x, y)
    else:
        plt.plot(y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    if save:
        if file_name is not None:
            plt.savefig(file_name)
        else:
            date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'{date_time}.png')


def find_discontinuity_points(data, dx=0.01):
    dy = diff(data.reshape(100, )) / dx
    # plt.figure()
    # plt.scatter(data, color='red')
    # plt.plot(dy)
    # plt.title('Sorted weights')

    points = np.argwhere(dy > 1)
    return points.reshape(points.shape[0], )


def assign_weights(labels, clusters):
    """Convert the labels to their weights based on the clusters array"""

    for cluster in range(np.size(clusters)):
        idx = np.where(labels == cluster)
        labels[idx] = clusters[cluster]

    return labels


def get_clusters(models, n_clusters, ranges, **kwargs):
    """
    Return an array of size n_clusters of all weights
    TODO: add kwargs for degree of PolynomialFeatures
    """
    clusters = np.zeros(n_clusters)

    for n_cluster in range(n_clusters):
        idx = search_range(n_clusters, ranges)
        x = np.array([n_cluster]).reshape(1,-1)
        if 'degree' in kwargs:
            poly_reg = PolynomialFeatures(degree=kwargs['degree'])
            x = poly_reg.fit_transform(np.int32(idx).reshape(1, -1))

        clusters[n_cluster] = models[idx].predict(x)

    return clusters


def get_models(x, y, disc, **kwargs):
    """Return a list of linear regression models"""
    models = []

    for idx in disc:
        x_slice = x[idx[0]:idx[1] + 1]
        y_slice = y[idx[0]:idx[1] + 1]
        if 'degree' in kwargs:
            poly_reg = PolynomialFeatures(degree=kwargs['degree'])
            x_slice = poly_reg.fit_transform(x[idx[0]:idx[1] + 1])

        lin_reg = LinearRegression().fit(x_slice, y_slice)
        print(r2_score(y_slice, lin_reg.predict(x_slice)))

        models.append(lin_reg)

    return models

