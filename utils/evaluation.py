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

def acc_tf(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
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
    
    '''
    # Split data in kmeans and FHC
    y_pred_s = y_pred[:int(len(y_pred) / 2)].astype(int)
    y_pred_t = y_pred[int(len(y_pred) / 2):].astype(int)
    y_pred_test_s = y_pred_test[:int(len(y_pred_test) / 2)].astype(int)
    y_pred_test_t = y_pred_test[int(len(y_pred_test) / 2):].astype(int)

    train_acc_k = acc(y_true.astype(int), y_pred_s)
    train_acc_f = acc(y_true.astype(int), y_pred_t)
    test_acc_k = acc(y_true_test.astype(int), y_pred_test_s)
    test_acc_f = acc(y_true_test.astype(int), y_pred_test_t)
    '''
    
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


