# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import linear_sum_assignment
import wandb


def acc(y_true, y_pred):
    """
    Calculate the accuracy of clustering results by finding the best match
    between the cluster labels and the true labels.

    Args:
        y_true (numpy.ndarray): The true labels, numpy array with shape `(n_samples,)`.
        y_pred (numpy.ndarray): The predicted labels, numpy array with shape `(n_samples,)`.

    Returns:
        float: The accuracy of the clustering, a value between 0 and 1.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def acc_tf(y_true, y_pred):
    """
    Calculate clustering accuracy for TensorFlow tensors. Requires scikit-learn installed.

    Args:
        y_true (tf.Tensor): The true labels, TensorFlow tensor with shape `(n_samples,)`.
        y_pred (tf.Tensor): The predicted labels, TensorFlow tensor with shape `(n_samples,)`.

    Returns:
        float: The accuracy of the clustering, a value between 0 and 1.
    """
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    y_pred = y_pred.argmax(1)
    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def evaluate_clustering(y_pred, y_true, y_pred_test=None, y_true_test=None):
    """
    Evaluate the clustering performance and log the results using WandB.

    Args:
        y_pred (numpy.ndarray): Predicted labels for the training set.
        y_true (numpy.ndarray): True labels for the training set.
        y_pred_test (numpy.ndarray, optional): Predicted labels for the test set.
        y_true_test (numpy.ndarray, optional): True labels for the test set.

    Returns:
        tuple: A tuple containing training accuracy and test accuracy (if available).
    """

    train_acc = acc(y_true.astype(int), y_pred)
    test_acc = np.nan
    if y_pred_test is not None:
        test_acc = acc(y_true_test.astype(int), y_pred_test)
        # Log with WandB
        wandb.log({"Final Train ACC": train_acc,
                   "Final Test ACC": test_acc})
    else:
        wandb.log({"Final Train ACC": train_acc})

    return train_acc, test_acc
