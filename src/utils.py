import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
from sklearn.metrics import auc
def get_adj_from_uv(u, v, node_num):
    adj = np.zeros([node_num, node_num])
    for i in range(len(u)):
        adj[u[i], v[i]] = 1
    return adj

def norm_adj(adj):
    I = torch.eye(adj.shape[0])
    adj_hat = I + adj
    D = torch.diag((torch.sum(adj_hat, dim=1)+1).pow(-0.5))
    adj_norm = D @ adj_hat @ D
    return adj_norm

def get_random_mask_ogb(features, r):
    probs = torch.full(features.shape, 1 / r)
    mask = torch.bernoulli(probs)
    return mask

def get_prauc(true_labels, pred_scores):
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    return auc(recall, precision)

def get_rocauc(true_labels, pred_scores):
    return roc_auc_score(true_labels, pred_scores)

def get_f1_score(true_labels, pred_scores):
    precision, recall, threshold = precision_recall_curve(true_labels, pred_scores)
    diff = np.abs(precision - recall)
    best_threshold = threshold[np.argmin(diff)]
    pred_labels = np.where(pred_scores > best_threshold, 1, 0)

    return f1_score(true_labels, pred_labels)

def perturbd_edges(u, v, ratio):
    del_num = int(len(u) * ratio)
    np.random.seed(0)
    del_idx = np.random.choice(len(u), del_num, replace=False)
    perturbed_u = np.delete(u, del_idx)
    perturbed_v = np.delete(v, del_idx)
    return perturbed_u, perturbed_v