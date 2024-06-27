import joblib
import numpy as np
import pandas as pd
import torch
from utils import *
from model import *
from torch import nn
from sklearn.model_selection import KFold
import sys
import copy
import argparse

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def self_semi_train_loss(features, mask_ratio, adj, model_s):
    mask = get_random_mask_ogb(features, mask_ratio).to(device)
    masked_features = features * (1 - mask)
    logits, Adj = model_s(masked_features, adj)
    indices = mask > 0
    loss_r = F.mse_loss(logits[indices], features[indices], reduction='mean')
    
    return loss_r, Adj

def train(data, args):
    mask_ratio = args.r
    cv_num = args.cv_num
    EPOCH = args.epochs
    features, labels, known_idx = data["features"], data["labels"], data["known_idx"]
    all_test_prauc = []
    all_test_rocauc = []
    all_test_f1_score = []

    for trial in range(0, cv_num):
        kf = KFold(n_splits=args.fold_num, random_state=trial, shuffle=True)
        print("-------cv: {} -------".format(trial))
        for train_val_idx_idx, test_idx_idx in kf.split(known_idx):
            best_val_prauc = 0
            best_test_prauc = 0
            best_test_rocauc = 0
            best_test_f1_score = 0
            train_mask, val_mask, test_mask = divide_bio_data(known_idx[train_val_idx_idx], known_idx[test_idx_idx], len(labels), trial)
            model_r = GCN_R(args.nlayers_r, features.shape[1], args.hidden_r, features.shape[1], args.dropout_r, args.dropout_r_adj).to(device)
            model_c = GCN_C(args.nlayers_c, features.shape[1], args.hidden_c, 2,  args.dropout_c, args.dropout_c_adj).to(device)
            adj = nn.Parameter(data["adj"].clone())
            optimizer_c = torch.optim.Adam(model_c.parameters(), lr=args.lr_c, weight_decay=args.w_decay_c)
            optimizer_r = torch.optim.Adam(model_r.parameters(), lr=args.lr_r, weight_decay=args.w_decay_r)
            optimizer_adj = torch.optim.Adam([adj], lr=args.lr_adj, weight_decay=args.w_decay_adj)
            for e in range(1, EPOCH + 1):
                model_r.train()
                model_c.train()
                loss_r, adj_s = self_semi_train_loss(features, mask_ratio, adj, model_r)
                logits = model_c(features, adj_s)
                loss_c = F.cross_entropy(logits[train_mask], labels[train_mask])
                loss = loss_c + loss_r * args.lambda_
                optimizer_c.zero_grad()
                optimizer_r.zero_grad()
                optimizer_adj.zero_grad()
                loss.backward()
                optimizer_c.step()
                optimizer_r.step()
                optimizer_adj.step()

                model_c.eval()
                model_r.eval()
                logits_test = model_c(features, adj_s.detach())
                pred_score_test = F.softmax(logits_test, dim=1)[:, 1]
                train_prauc = get_prauc(labels.cpu().detach().numpy()[train_mask], pred_score_test.cpu().detach().numpy()[train_mask])
                val_prauc = get_prauc(labels.cpu().detach().numpy()[val_mask], pred_score_test.cpu().detach().numpy()[val_mask])
                test_prauc = get_prauc(labels.cpu().detach().numpy()[test_mask], pred_score_test.cpu().detach().numpy()[test_mask])
                test_rocauc = get_rocauc(labels.cpu().detach().numpy()[test_mask], pred_score_test.cpu().detach().numpy()[test_mask])
                test_f1_score = get_f1_score(labels.cpu().detach().numpy()[test_mask], pred_score_test.cpu().detach().numpy()[test_mask])
                if best_val_prauc < val_prauc:
                    best_val_prauc = val_prauc
                    best_test_prauc = test_prauc
                    best_test_rocauc = test_rocauc
                    best_test_f1_score = test_f1_score
                    

                    
                if e % EPOCH == 0:
                    print('In epoch {}, loss_c: {:.4f}, train prauc: {:.3f}, val prauc: {:.3f} (best {:.3f}), test prauc: {:.3f} (best {:.3f})'.format(e, loss_c, train_prauc, val_prauc, best_val_prauc, test_prauc, best_test_prauc))
            all_test_prauc.append(best_test_prauc)
            all_test_rocauc.append(best_test_rocauc)
            all_test_f1_score.append(best_test_f1_score)
        print("now prauc:{:.3f}; now rocauc:{:.3f}; now f1_score:{:.3f}".format(np.mean(all_test_prauc), np.mean(all_test_rocauc), np.mean(all_test_f1_score)))
        sys.stdout.flush()
    print("mean of prauc is {}".format(np.mean(all_test_prauc)))
    print("mean of rocauc is {}".format(np.mean(all_test_rocauc)))
    print("mean of f1 score is {}".format(np.mean(all_test_f1_score)))

def test(data):
    mask_ratio = args.r
    EPOCH = args.epochs
    features, labels, known_idx = data["features"], data["labels"], data["known_idx"]
    random_seed = 0
    kf = KFold(n_splits=5, random_state=random_seed, shuffle=True)
    for train_val_idx_idx, test_idx_idx in kf.split(known_idx):
        best_val_prauc = 0
        train_mask, val_mask, test_mask = divide_bio_data(known_idx[train_val_idx_idx], known_idx[test_idx_idx], len(labels), random_seed)
        train_mask = train_mask | test_mask
        model_r = GCN_R(args.nlayers_r, features.shape[1], args.hidden_r, features.shape[1], args.dropout_r,args.dropout_r_adj).to(device)
        model_c = GCN_C(args.nlayers_c, features.shape[1], args.hidden_c, 2, args.dropout_c, args.dropout_c_adj).to(device)
        adj = nn.Parameter(data["adj"].clone())
        optimizer_c = torch.optim.Adam(model_c.parameters(), lr=args.lr_c, weight_decay=args.w_decay_c)
        optimizer_s = torch.optim.Adam(model_r.parameters(), lr=args.lr_r, weight_decay=args.w_decay_r)
        optimizer_adj = torch.optim.Adam([adj], lr=args.lr_adj, weight_decay=args.w_decay_adj)
        for e in range(1, EPOCH + 1):
            model_r.train()
            model_c.train()
            loss_r, adj_s = self_semi_train_loss(features, mask_ratio, adj, model_r)
            logits = model_c(features, adj_s)
            loss_c = F.cross_entropy(logits[train_mask], labels[train_mask])
            loss = loss_c + loss_r * args.lambda_
            optimizer_c.zero_grad()
            optimizer_s.zero_grad()
            optimizer_adj.zero_grad()
            loss.backward()
            optimizer_c.step()
            optimizer_s.step()
            optimizer_adj.step()

            model_c.eval()
            model_r.eval()
            _, adj_s_test = self_semi_train_loss(features, mask_ratio, adj, model_r)
            logits_test = model_c(features, adj_s_test)
            pred_score_test = F.softmax(logits_test, dim=1)[:, 1]
            train_prauc = get_prauc(labels.cpu().detach().numpy()[train_mask], pred_score_test.cpu().detach().numpy()[train_mask])
            val_prauc = get_prauc(labels.cpu().detach().numpy()[val_mask], pred_score_test.cpu().detach().numpy()[val_mask])
            
            if best_val_prauc < val_prauc:
                best_val_prauc = val_prauc
                best_adj, best_model_c = copy.deepcopy(adj_s), copy.deepcopy(model_c)
                
            if e % EPOCH == 0:
                print('In epoch {}, loss_c: {:.4f}, train prauc: {:.3f}, val prauc: {:.3f} (best {:.3f}) '.format(e, loss_c, train_prauc, val_prauc, best_val_prauc))

        return best_adj, best_model_c

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=1000, help='The number of epochs to train.')
    parser.add_argument('-lr_c', type=float, default=1e-3, help='Learning rate of GCN_C.')
    parser.add_argument('-lr_r', type=float, default=1e-3, help='Learning rate of GCN_R.')
    parser.add_argument('-lr_adj', type=float, default=1e-7, help='Learning rate of adj.')
    parser.add_argument('-w_decay_c', type=float, default=2e-3, help='Weight decay of GCN_C.')
    parser.add_argument('-w_decay_r', type=float, default=0.0, help='Weight decay of GCN_R.')
    parser.add_argument('-w_decay_adj', type=float, default=0.0, help='Weight decay of adj.')
    parser.add_argument('-hidden_c', type=int, default=128, help='Hidden channel number of GCN_C.')
    parser.add_argument('-hidden_r', type=int, default=128, help='Hidden channel number of GCN_R.')
    parser.add_argument('-dropout_c', type=float, default=0.5, help='Dropout rate of GCN_C.')
    parser.add_argument('-dropout_r', type=float, default=0.5, help='Dropout rate of GCN_R.')
    parser.add_argument('-dropout_c_adj', type=float, default=0.5, help='Dropout rate of adj in GCN_C.')
    parser.add_argument('-dropout_r_adj', type=float, default=0.5, help='Dropout rate of adj in GCN_R.')
    parser.add_argument('-nlayers_c', type=int, default=3, help='The number of layers for GCN_C.')
    parser.add_argument('-nlayers_r', type=int, default=3, help='The number of layers for GCN_R.')
    parser.add_argument('-cv_num', type=int, default=10, help='The number of cross-validation times.')
    parser.add_argument('-fold_num', type=int, default=5, help='The number of fold number.')
    parser.add_argument('-lambda_', type=float, default=0.1, help='Hyperparameter to balance the loss.')
    parser.add_argument('-r', type=int, default=20, help='Mask ratio.')
    parser.add_argument('-removal_proportion', type=float, default=0.0, help='Removal proportion of edges in the network.')
    args = parser.parse_args()

    # load data
    network_u = joblib.load("joblib/network_u.joblib")
    network_v = joblib.load("joblib/network_v.joblib")
    features, nfeats, labels, nclasses, known_idx, unknown_idx = load_bio_data("data/omics_features.tsv")
    features = torch.from_numpy(features).to(torch.float32).to(device)
    labels = torch.tensor(labels).type(torch.int64).to(device)
    
    # train and evaluate
    ptb_network_u, ptb_network_v = perturbd_edges(network_u, network_v, args.removal_proportion)
    adj = torch.tensor(get_adj_from_uv(ptb_network_u, ptb_network_v, features.shape[0]))
    adj = norm_adj(adj).float().to(device)
    eval_data = {"features":features, "labels":labels, "known_idx":known_idx, "adj":adj}
    train(eval_data, args)

    # train and test
    adj = torch.tensor(get_adj_from_uv(network_u, network_v, features.shape[0]))
    adj = norm_adj(adj).float().to(device)
    test_data = {"features":features, "labels":labels, "known_idx":known_idx, "adj":adj}
    best_adj, best_model_c = test(test_data)
    joblib.dump(best_adj, "joblib/best_adj.joblib")
    joblib.dump(best_model_c, "joblib/best_model_c.joblib")

    omics_data = np.loadtxt("data/omics_features.tsv", delimiter='\t', skiprows=1, dtype=str)
    logits = best_model_c(features, best_adj)
    test_proba = F.softmax(logits, dim=1)[unknown_idx]
    gene_names = omics_data[:, 0][unknown_idx]

    PCDGs_DF = pd.DataFrame(test_proba.cpu().detach().numpy())
    PCDGs_DF.index = gene_names
    df_sorted = PCDGs_DF.sort_values(by=0)
    df_sorted.to_csv('result/PCDGs.csv', index=True)