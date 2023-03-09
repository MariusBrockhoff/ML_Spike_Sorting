# -*- coding: utf-8 -*-
# import hdbscan
from sklearn.cluster import KMeans, DBSCAN
import numpy as np


def clustering(data, method, n_clusters=5, eps=0.01, min_cluster_size=100):
    if method == "Kmeans":
        kmeans = KMeans(n_clusters=n_clusters, n_init=50)
        y_pred = kmeans.fit_predict(data)

    elif method == "DBSCAN":
        clusterer = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(data)
        y_pred = clusterer.labels_
        n_clusters = len(np.unique(y_pred))

    # elif method=="HDBSCAN":
    #     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    #     clusterer.fit(data)
    #     y_pred = clusterer.labels_
    #     n_clusters = len(np.unique(y_pred))

    else:
        raise ValueError("Please choose a valid clusering method! Chooose between Butter_bandpass, "
                         "Butter_highpass, Elliptic_bandpass or Elliptic_highpass")

    return y_pred, n_clusters
