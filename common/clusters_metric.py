import numpy as np
from sklearn.cluster import KMeans


class ClustersMetric(object):
    def __init__(self, training_data, n_clusters=2):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.kmeans.fit(training_data)
        self.cluster_center = self.kmeans.cluster_centers_.copy()

    def evaluate(self, gen_func):
        centers = np.expand_dims(self.cluster_center, axis=0)
        d = gen_func()
        distance = np.sqrt(np.sum(np.power(np.expand_dims(d.cpu().detach().numpy(), axis=1) - centers, 2.0), axis=2))
        distance_index = distance.argmin(axis=1)
        distance_close = distance[:, 0] * (1 - distance_index) + distance_index * distance[:, 1]
        return distance_close.mean()
