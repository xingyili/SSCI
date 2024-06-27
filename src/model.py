import sys
sys.path.append('gen_adj')
import torch
import torch.nn.functional as F
from dataloader import *
from torch import nn
from layer import *
from torch_geometric.nn import GCNConv, GATConv, ChebConv

class GCN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate=0.5,
                 **kwargs):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.GCN1 = GCNConv(self.input_dim, 128)
        self.GCN2 = GCNConv(128, 128)
        self.GCN3 = GCNConv(128, self.output_dim)

    def forward(self, features, adj): 
        x, edge_index = features, adj
        x = self.GCN1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        x = self.GCN2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        x = self.GCN3(x, edge_index)

        return F.log_softmax(x, dim=1)
class GAT_C(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate=0.5,
                 **kwargs):
        super(GAT_C, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.GCN1 = GATConv(self.input_dim, 128)
        self.GCN2 = GATConv(128, 128)
        self.GCN3 = GATConv(128, self.output_dim)

    def forward(self, features, adj):
        x, edge_index = features, adj
        x = self.GCN1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        x = self.GCN2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        x = self.GCN3(x, edge_index)

        return F.log_softmax(x, dim=1)
    
class Cheb_C(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate=0.5,
                 **kwargs):
        super(Cheb_C, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.GCN1 = ChebConv(self.input_dim, 128, K=2, normalization="sym")
        self.GCN2 = ChebConv(128, 128, K=2, normalization="sym")
        self.GCN3 = ChebConv(128, self.output_dim, K=2, normalization="sym")

    def forward(self, features, adj):
        x, edge_index = features, adj
        x = self.GCN1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        x = self.GCN2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        x = self.GCN3(x, edge_index)

        return F.log_softmax(x, dim=1)

class GCN_R(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, nclasses, dropout, dropout_adj):
        super(GCN_R, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(GCNConv_dense(in_dim, hidden_dim))
        for i in range(nlayers - 2):
            self.layers.append(GCNConv_dense(hidden_dim, hidden_dim))
        self.layers.append(GCNConv_dense(hidden_dim, nclasses))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj

    def forward(self, x, Adj_):
        Adj = self.dropout_adj(Adj_)
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x, Adj_

class GCN_C(nn.Module):
    def __init__(self, nlayers, in_channels, hidden_channels, out_channels, dropout, dropout_adj):
        super(GCN_C, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(GCNConv_dense(in_channels, hidden_channels))
        for i in range(nlayers - 2):
            self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
        self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj

    def forward(self, x, adj_t):
        Adj = self.dropout_adj(adj_t)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x