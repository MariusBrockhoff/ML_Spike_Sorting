# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import KDTree


def fhc_lpd(data, k, C):
    """
        Performs clustering using a Fast Hierarchical Clustering with Local Peak Density (FHC_LPD).

        Args:
            data (numpy.ndarray): The dataset to cluster, where each row represents an observation.
            k (int): The number of neighbors to consider for each point.
            C (int): The desired number of clusters.

        Returns:
            numpy.ndarray: An array of cluster labels for each observation in the dataset.
        """
    # Normalization
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    data[np.isnan(data)] = 0

    # Fast KNN based on kd-tree (efficient for lower dimensions)
    n, d = data.shape
    if d <= 10:
        tree = KDTree(data)
        knn_dist, knn = tree.query(data, k=k)
    else:
        """#Faster but High RAM version
        dist = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2))
        knn = np.argsort(dist, axis=1)[:, 1:k+1]
        knn_dist = dist[np.arange(n)[:, None], knn]"""

        # Slower but less memory hungry
        dist = np.empty((n, n), dtype=np.float64)
        knn = np.empty((n, k), dtype=np.int64)
        fills = np.empty((n, d), dtype=np.float64)
        for i in range(n):
            # Compute squared Euclidean distances to all other points
            np.square(data[i] - data, out=fills)
            np.sum(fills, axis=1, out=dist[i])

            # Find indices of k+1 nearest neighbors
            neighbors = np.argpartition(dist[i], k + 1)[:k + 1]
            knn[i] = neighbors[1:]  # exclude the point itself

            # Replace distances to self with infinity
            dist[i, i] = np.inf
        # Compute distances to k nearest neighbors
        knn_dist = np.sqrt(dist[np.arange(n)[:, None], knn])
        for i, neighbors in enumerate(knn):
            knn_dist[i] = dist[i, neighbors]

    # calculate the knn-density value
    rho = knn_dist[:, -1] ** -1

    # search for NPN(neighbor-parent node) and record depth value
    OrdRho = np.argsort(-rho)
    omega = np.zeros(n, dtype=int)
    NPN = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(1, k):
            neigh = knn[OrdRho[i], j]
            if rho[OrdRho[i]] < rho[neigh]:
                NPN[OrdRho[i]] = neigh
                omega[OrdRho[i]] = omega[neigh] + 1
                break

    # find sub-cluster centers(namely local density peaks)
    sub_centers = np.where(omega == 0)[0]
    n_sc = len(sub_centers)

    # generate sub-clusters
    sub_L = -1 * np.ones(n, dtype=int)
    sub_L[sub_centers] = np.arange(n_sc)
    for i in range(n):
        if sub_L[OrdRho[i]] == -1:
            sub_L[OrdRho[i]] = sub_L[NPN[OrdRho[i]]]

    # calculate center-association degree PHI
    lambda_val = 0.9
    AAA = lambda_val ** np.arange(n)
    PHI = AAA[omega]

    # calculate SIM(similarity) matrix between sub-clusters
    PHIMatrix = np.zeros((n_sc, n_sc))
    for i in range(n):
        for j in range(1, k):
            jj = knn[i, j]
            PHISum = PHI[jj] + PHI[i]
            if sub_L[i] != sub_L[jj] and PHIMatrix[sub_L[i], sub_L[jj]] < PHISum:
                if i in knn[jj, 1:k]:
                    PHIMatrix[sub_L[i], sub_L[jj]] = PHISum
                    PHIMatrix[sub_L[jj], sub_L[i]] = PHISum

    SIM = np.zeros((n_sc, n_sc))
    SIM_list = []
    for i in range(n_sc - 1):
        for j in range(i + 1, n_sc):
            if PHIMatrix[i, j] > 0 and PHIMatrix[j, i] > 0:
                SIM[i, j] = PHIMatrix[i, j] / 2  # InterPenetration of cl=i and cl=j
            SIM_list.append(SIM[i, j])

    # SingleLink clustering of sub-clusters according to SIM
    SingleLink = linkage(1 - np.array(SIM_list), method='single')

    F_sub_L = fcluster(SingleLink, t=C, criterion='maxclust')

    # Assign final cluster label
    CL = np.zeros(len(data))
    for i in range(n_sc):
        AA = np.where(sub_L == i)[0]
        CL[AA] = F_sub_L[i]
    return CL


def clustering(data, method, n_clusters=5, eps=0.01, min_cluster_size=100, knn=1000):
    """
        Performs clustering on the given dataset using specified methods.

        Args:
            data (numpy.ndarray): The dataset to cluster.
            method (str): The clustering method to use. Can be 'Kmeans', 'DBSCAN', 'FHC_LPD', or 'Kmeans_FHC_LPD'.
            n_clusters (int, optional): The number of clusters for Kmeans and FHC_LPD. Default is 5.
            eps (float, optional): Epsilon parameter for DBSCAN, specifies how close points should be to each other to
                                   be considered a part of a cluster. Default is 0.01.
            min_cluster_size (int, optional): The minimum number of samples in a neighborhood for a point to be
                                              considered as a core point in DBSCAN. Default is 100.
            knn (int, optional): The number of nearest neighbors to consider for FHC_LPD. Default is 1000.
        Returns:
            tuple: A tuple containing the array of cluster labels and the number of clusters detected. The array assigns
                   each observation in the dataset to a cluster.

        Raises:
            ValueError: If an invalid clustering method is chosen.
    """
    if method == "Kmeans":
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(data)

    elif method == "DBSCAN":
        clusterer = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(data)
        y_pred = clusterer.labels_
        n_clusters = len(np.unique(y_pred))

    elif method == "FHC_LPD":
        y_pred = fhc_lpd(data, k=knn, C=n_clusters)

    elif method == "Kmeans_FHC_LPD":
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        y_pred_k = kmeans.fit_predict(data)
        y_pred = fhc_lpd(data, k=knn, C=n_clusters)
        y_pred = np.concatenate((y_pred_k, y_pred), axis=0)

    else:
        raise ValueError("Please choose a valid clusering method! Chooose between Butter_bandpass, "
                         "Butter_highpass, Elliptic_bandpass or Elliptic_highpass")

    return y_pred, n_clusters
