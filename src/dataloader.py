import numpy as np
import joblib
from sklearn.model_selection import train_test_split

def load_bio_data(dataset):
    data = np.loadtxt(dataset, delimiter='\t', skiprows=1, dtype=str)
    labels = data[:, -1].astype(np.float32).astype(np.int32)
    features = data[:, 1:-1].astype(np.float32)
    nfeats = features.shape[1]
    nclasses = labels.max() + 1
    X_p_index = joblib.load("joblib/p_idx.joblib").astype(np.int32)
    X_n_index = joblib.load("joblib/n_idx.joblib").astype(np.int32)
    X_u_index = joblib.load("joblib/u_idx.joblib").astype(np.int32)
    return features, nfeats, labels, nclasses, np.concatenate((X_p_index, X_n_index)), X_u_index

def divide_bio_data(train_val_idx, test_idx, mask_len, random_num):
    train_idx, val_idx, _, _ = train_test_split(train_val_idx, train_val_idx, test_size=0.125, random_state=random_num)
    train_mask, val_mask, test_mask = np.array([False] * mask_len), np.array([False] * mask_len), np.array([False] * mask_len)
    for i in range(mask_len):
        if i in train_idx:
            train_mask[i] = True
        if i in val_idx:
            val_mask[i] = True
        if i in test_idx:
            test_mask[i] = True
        if (i in train_idx and i in val_idx) or (i in train_idx and i in test_idx) or (i in val_idx and i in test_idx):
            print("data divide error!")
    return train_mask, val_mask, test_mask