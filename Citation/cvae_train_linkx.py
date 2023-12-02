from torch_geometric.datasets import TUDataset
from tqdm import tqdm, trange
import argparse
import os.path as osp
import time
import numpy as np
import copy
import gc

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as gnnDataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch.utils.data import TensorDataset, DataLoader as nnDataLoader, RandomSampler
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_add_pool
from tqdm.auto import tqdm, trange

from cvae_pretrain import loss_fn, feature_tensor_normalize
from cvae_models import VAE
from gin.models import GIN, LAGIN

from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.datasets as gnnDatasets
import torch_geometric.nn as gnn

from torch_geometric.loader import DataLoader as gnnDataLoader

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='reed98')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--pretrain_epochs', type=int, default=10)
parser.add_argument('--latent_size', type=int, default=10)
parser.add_argument('--num_models', type=int, default=50)
parser.add_argument('--warmup', type=int, default=100)
parser.add_argument('--update_epochs', type=int, default=20)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = gnnDatasets.LINKXDataset(root="/content/datasets/", name=args.dataset)


idx_train, idx_testval = train_test_split(range(len(dataset.x)), test_size=0.7, stratify=dataset.y, random_state=20)
idx_test, idx_val = train_test_split(idx_testval, test_size=0.5)

idx_train, idx_test, idx_val = torch.LongTensor(idx_train), torch.LongTensor(idx_test), torch.LongTensor(idx_val) 

adj = to_scipy_sparse_matrix(dataset.edge_index).tocsr()
adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
features = dataset.x
features_normalized = feature_tensor_normalize(features)
labels = (dataset.y + 1).long()
idx_train = torch.LongTensor(idx_train)


# making features
x_list, c_list = [], []
for i in trange(adj.shape[0]):
    x = features[adj[[i], :].nonzero()[1]]
    c = np.tile(features[i], (x.shape[0], 1))
    x_list.append(x)
    c_list.append(c)
features_x = np.vstack(x_list)
features_c = np.vstack(c_list)
del x_list
del c_list
gc.collect()


features_x = torch.tensor(features_x, dtype=torch.float32)
features_c = torch.tensor(features_c, dtype=torch.float32)

cvae_features = torch.tensor(features, dtype=torch.float32).to(device)

cvae_dataset = TensorDataset(features_x, features_c)
cvae_dataset_sampler = RandomSampler(cvae_dataset)
cvae_dataset_dataloader = nnDataLoader(cvae_dataset, sampler=cvae_dataset_sampler, batch_size=64)
from gcn.models import GCN

hidden = 32
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
epochs = 200

model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            nclass=labels.max().item() + 1,
            dropout=dropout)

model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

model.to(device)
features_normalized = features_normalized.to(device)
adj_normalized = adj_normalized.to(device)
cvae_features = cvae_features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)

for _ in range(int(epochs / 2)):
    model.train()
    model_optimizer.zero_grad()
    output = model(features_normalized, adj_normalized)
    output = torch.log_softmax(output, dim=1)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    model_optimizer.step()


# Pretrain
cvae = VAE(
    encoder_layer_sizes=[features.shape[1], 256],
    latent_size=args.latent_size,
    decoder_layer_sizes=[256, features.shape[1]],
    conditional=True,
    conditional_size=features.shape[1]
).to(device)
cvae_optimizer = torch.optim.Adam(cvae.parameters(), lr=args.pretrain_lr)

# Pretrain
t = 0
best_augmented_features  = None
cvae_model = None
best_score = -float("inf")
for _ in trange(args.pretrain_epochs, desc='Run CVAE Train'):
    pbar = tqdm(cvae_dataset_dataloader, ncols=850)
    for _, (x, c) in enumerate(pbar):
        cvae.train()
        x, c = x.to(device), c.to(device)
        if args.conditional:
            recon_x, mean, log_var, _ = cvae(x, c)
        else:
            recon_x, mean, log_var, _ = cvae(x)
        cvae_loss = loss_fn(recon_x, x, mean, log_var)

        cvae_optimizer.zero_grad()
        cvae_loss.backward()
        cvae_optimizer.step()


        z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
        augmented_features = cvae.inference(z, cvae_features)
        augmented_features = feature_tensor_normalize(augmented_features).detach()

        total_logits = 0
        cross_entropy = 0
        for i in range(args.num_models):
            logits = model(augmented_features, adj_normalized)
            total_logits += F.softmax(logits, dim=1)
            output = F.log_softmax(logits, dim=1)
            cross_entropy += F.nll_loss(output[idx_train], labels[idx_train])
        output = torch.log(total_logits / args.num_models)
        U_score = F.nll_loss(output[idx_train], labels[idx_train]) - cross_entropy / args.num_models
        t += 1
        pbar.set_postfix_str(f"U Score: {U_score.item():.4f}, Best Score: {torch.tensor(best_score).item():.4f}")
        if U_score > best_score:
            best_score = U_score
            if t > args.warmup:
                cvae_model = copy.deepcopy(cvae)
                print("U_score: ", U_score, " t: ", t)
                best_augmented_features = copy.deepcopy(augmented_features)
                for i in range(args.update_epochs):
                    model.train()
                    model_optimizer.zero_grad()
                    output = model(best_augmented_features, adj_normalized)
                    output = torch.log_softmax(output, dim=1)
                    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                    loss_train.backward()
                    model_optimizer.step()

if cvae_model==None : cvae_model = copy.deepcopy(cvae)

torch.save(cvae_model, "model/{args.dataset}.pkl")
