import torch_geometric.nn as gnn
import argparse
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as gnnDataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_add_pool


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)

class LAGIN(nn.Module):
    def __init__(self, concat, nfeat, nhid, out_ch, num_layers):
        super().__init__()

        self.concat = concat
        self.nfeat, self.nhid = nfeat, nhid
        self.convs = nn.ModuleList()
        for i in range(concat):
            convs = nn.ModuleList()
            nfeat, nhid = self.nfeat, self.nhid
            for _ in range(num_layers):
                mlp = gnn.MLP([nfeat, nhid, nhid])
                convs.append(gnn.GINConv(nn=mlp, train_eps=False))
                nfeat = nhid
            self.convs.append(convs)

        self.mlp = gnn.MLP([nhid*concat, nhid, out_ch],
                       norm=None, dropout=0.5)

    def forward(self, data):
        hidden = []
        for k, convs in enumerate(self.convs):
            x, edge_index, batch = data[k].x.clone(), data[k].edge_index.clone(), data[k].batch.clone()
            for conv in convs:
                x = conv(x, edge_index).relu()
            x = gnn.global_add_pool(x, batch)
            hidden.append(x)
        x = torch.cat((hidden), dim=-1)
        return self.mlp(x)