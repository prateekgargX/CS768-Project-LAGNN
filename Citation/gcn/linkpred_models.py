import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LPLAGCN(torch.nn.Module):
    def __init__(self, concat, nfeat, nhid, out_ch, dropout):
        super().__init__()

        self.concat = concat
        self.conv1 = nn.ModuleList([GCNConv(nfeat, nhid) for _ in range(concat)])
        self.conv2 = GCNConv(concat*nhid, out_ch)
        self.dropout = dropout

    def encode(self, data):
        hidden = []
        for k, conv in enumerate(self.conv1):
            x, edge_index = data[k].x.to(device), data[k].edge_index.to(device)
            x = F.dropout(x, self.dropout, self.training)
            hidden.append(conv(x, edge_index).relu())
        x = torch.cat((hidden), dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
