from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import copy
import random
import torch.nn.functional as F
import torch.optim as optim
import cvae_pretrain
import os

from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
from gcn.linkpred_models import LAGCN
from tqdm import trange

import torch_geometric.datasets as gnnDatasets
import torch_geometric.nn as gnn

from torch_geometric.loader import DataLoader as gnnDataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix

from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

exc_path = sys.path[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=4)
parser.add_argument("--concat", type=int, default=4)
parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')

parser.add_argument('--dataset', default='cora', help='Dataset string, only Planetoid and LINKXDataset names supported')
parser.add_argument('--pretrained_model', default='vae', help='pretrained model to use')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')


args = parser.parse_args()

concat_ = args.concat

linkxnames = ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]
if args.dataset in linkxnames:
    dataset = gnnDatasets.LINKXDataset(root="./data/", name=args.dataset)
    linkx = True
else:
    dataset = gnnDatasets.Planetoid(root='./data/', name=args.dataset)
    linkx = False

if args.pretrained_model == "vae":
    cvae_model = torch.load("{}/model/{}.pkl".format(exc_path, args.dataset))
if args.pretrained_model == "nf":
    cvae_model = torch.load("{}/model_CNF/{}.pkl".format(exc_path, args.dataset))

if linkx:
    from sklearn.model_selection import train_test_split

    idx_train, idx_testval = train_test_split(range(len(dataset.x)), test_size=0.7, stratify=dataset.y, random_state=20)
    idx_test, idx_val = train_test_split(idx_testval, test_size=0.5)

    adj = to_scipy_sparse_matrix(dataset.edge_index).tocsr()
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
    features = dataset.x
    features_normalized = cvae_pretrain.feature_tensor_normalize(features)
    labels = (dataset.y + 1).long()
    idx_train = torch.LongTensor(idx_train)
    idx_val, idx_test = torch.LongTensor(idx_val), torch.LongTensor(idx_test)

    adj_normalized = adj_normalized.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    features_normalized = features_normalized.to(device)

    dataset = gnnDatasets.LINKXDataset(
        root="./data/", name=args.dataset,
        transform=T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                        add_negative_train_samples=False)
    )

    train_data, val_data, test_data = dataset[0]
    train_data.x = features_normalized.clone()
    val_data.x = features_normalized.clone()
    test_data.x = features_normalized.clone()

else:
    # Load data
    adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)

    # Normalize adj and features
    features = features.toarray()
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    features_normalized = normalize_features(features)

    # To PyTorch Tensor
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    features_normalized = torch.FloatTensor(features_normalized)
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    adj_normalized = adj_normalized.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    features_normalized = features_normalized.to(device)

    dataset = Planetoid(
        "./data/", name=args.dataset,
        transform=T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                        add_negative_train_samples=False)
    )

    train_data, val_data, test_data = dataset[0]
    train_data.x = features_normalized.clone()
    val_data.x = features_normalized.clone()
    test_data.x = features_normalized.clone()



def get_augmented_features(concat):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features)
        augmented_features = cvae_pretrain.feature_tensor_normalize(augmented_features).detach()
        if args.cuda:
            X_list.append(augmented_features.to(device))
        else:
            X_list.append(augmented_features)
    return X_list


@torch.no_grad()
def test(data, concat):
    model.eval()
    xlist = get_augmented_features(concat) + [features_normalized]
    dataclones = [data.clone() for x in xlist]
    for i in range(len(xlist)):
        dataclones[i].x = xlist[i]
    z = model.encode(dataclones)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())



# AUGMENTED
from tqdm.auto import tqdm

args.concat = concat_
n_epochs = args.epochs
num_runs = args.runs

test_res = []
for run in range(1,num_runs+1):
    model = LAGCN(
        args.concat+1,
        nfeat=features.shape[1],
        nhid=256,
        out_ch=64,
        dropout=0.2
    ).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4, amsgrad=True)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = final_test_auc = 0
    pbar = tqdm(range(1, n_epochs+1), leave=False)
    for epoch in pbar:
        pbar.set_description_str(f"(Run {run}) Epoch: {epoch:3d}|{n_epochs}")
        model.train()
        output_list = []
        for k in range(int(args.samples)):
            X_list = get_augmented_features(args.concat)
            X_list = X_list + [features_normalized]
            dataclones = [train_data.clone() for x in X_list]
            for i in range(len(X_list)):
                dataclones[i].x = X_list[i]

            output_list.append(model.encode(dataclones))

        loss = 0
        for n, output in enumerate(output_list):
            neg_edge_index = negative_sampling(
                edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
                num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

            edge_label_index = torch.cat(
                [train_data.edge_label_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                train_data.edge_label,
                train_data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0).to(device)

            out = model.decode(output, edge_label_index).view(-1)
            loss += criterion(out, edge_label)/len(output_list)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_auc = test(val_data, args.concat)
        test_auc = test(test_data, args.concat)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        if epoch % 5==0:
            pbar.set_postfix_str(f'loss: {loss:.4f}, val: {val_auc:.4f}, '
            f'test: {test_auc:.4f}')
    test_res.append(final_test_auc)
# print(f'Final Test: {final_test_auc:.4f}')

print(f"[AUGMENTED] test_mu: {np.mean(test_res):.4f}, test_std: {np.std(test_res):.4f} (over {num_runs} runs)")

# Unaugmented
args.concat = 0

test_res = []
for run in range(1,num_runs+1):
    model = LAGCN(
        args.concat+1,
        nfeat=features.shape[1],
        nhid=256,
        out_ch=64,
        dropout=0.2
    ).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4, amsgrad=True)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = final_test_auc = 0
    pbar = tqdm(range(1, n_epochs+1), leave=False)
    for epoch in pbar:
        pbar.set_description_str(f"(Run {run}) Epoch: {epoch:3d}|{n_epochs}")
        model.train()
        output_list = []
        for k in range(int(args.samples)):
            X_list = get_augmented_features(args.concat)
            X_list = X_list + [features_normalized]
            dataclones = [train_data.clone() for x in X_list]
            for i in range(len(X_list)):
                dataclones[i].x = X_list[i]

            output_list.append(model.encode(dataclones))

        loss = 0
        for n, output in enumerate(output_list):
            neg_edge_index = negative_sampling(
                edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
                num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

            edge_label_index = torch.cat(
                [train_data.edge_label_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                train_data.edge_label,
                train_data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0).to(device)

            out = model.decode(output, edge_label_index).view(-1)
            loss += criterion(out, edge_label)/len(output_list)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_auc = test(val_data, args.concat)
        test_auc = test(test_data, args.concat)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        if epoch % 5==0:
            pbar.set_postfix_str(f'loss: {loss:.4f}, val: {val_auc:.4f}, '
            f'test: {test_auc:.4f}')
    test_res.append(final_test_auc)

print(f"[UNAUGMENTED] test_mu: {np.mean(test_res):.4f}, test_std: {np.std(test_res):.4f} (over {num_runs} runs)")