# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import linear_sum_assignment
import wandb


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
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


def evaluate_clustering(y_pred, y_pred_test, y_true, y_true_test):
    # Split data in kmeans and FHC
    y_pred_s = y_pred[:int(len(y_pred) / 2), :]
    y_pred_t = y_pred[int(len(y_pred) / 2):, :]
    y_pred_test_s = y_pred_test[:int(len(y_pred_test) / 2), :]
    y_pred_test_t = y_pred_test[int(len(y_pred_test) / 2):, :]

    train_acc_k = acc(y_true, y_pred_s)
    train_acc_f = acc(y_true, y_pred_t)
    test_acc_k = acc(y_true_test, y_pred_test_s)
    test_acc_f = acc(y_true_test, y_pred_test_t)

    # Log with WandB
    wandb.log({
        "Train ACC Kmeans": train_acc_k,
        "Test ACC Kmeans": test_acc_k,
        "Train ACC FHC-LPD": train_acc_f,
        "Test ACC FHC-LPD": test_acc_f})
    train_acc = [train_acc_k, train_acc_f]
    test_acc = [test_acc_k, test_acc_f]
    return train_acc, test_acc


def DINO_evaluate_clustering(y_pred, y_pred_test, y_true, y_true_test):
    # Split data in student and teacher
    y_pred_s = y_pred[:int(len(y_pred) / 2), :]
    y_pred_t = y_pred[int(len(y_pred) / 2):, :]
    y_pred_test_s = y_pred_test[:int(len(y_pred_test) / 2), :]
    y_pred_test_t = y_pred_test[int(len(y_pred_test) / 2):, :]

    train_acc_s = acc(y_true, y_pred_s)
    train_acc_t = acc(y_true, y_pred_t)
    test_acc_s = acc(y_true_test, y_pred_test_s)
    test_acc_t = acc(y_true_test, y_pred_test_t)

    #Log with WandB
    wandb.log({
            "Train ACC Student": train_acc_s,
            "Test ACC Student": test_acc_s,
            "Train ACC Student": train_acc_t,
            "Test ACC Student": test_acc_t})
    train_acc = [train_acc_s, train_acc_t]
    test_acc = [train_acc_t, test_acc_t]
    return train_acc, test_acc