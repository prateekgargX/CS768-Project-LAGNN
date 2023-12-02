from torch_geometric.datasets import TUDataset
from tqdm import tqdm, trange
import argparse
import os.path as osp
import time
import numpy as np
import copy

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as gnnDataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch.utils.data import TensorDataset, DataLoader as nnDataLoader, RandomSampler
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_add_pool
from tqdm.auto import tqdm, trange

import gc

from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
from cvae_models import VAE
from cvae_pretrain import loss_fn, feature_tensor_normalize
from gcn.gin_models import GIN, LAGIN


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.01)
# parameters for training loop
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--concat', type=int, default=3)
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--samples', type=int, default=4)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = TUDataset("/content/datasets/", name=args.dataset).shuffle()
train_loader = gnnDataLoader(dataset[:0.9], args.batch_size, shuffle=True)
test_loader = gnnDataLoader(dataset[0.9:], args.batch_size)

big_graph = next(iter(gnnDataLoader(dataset, len(dataset))))
adj = to_scipy_sparse_matrix(big_graph.edge_index).tocsr()
features = big_graph.x
labels = big_graph.y
batch = big_graph.batch
features_normalized = feature_tensor_normalize(big_graph.x)
idx_train = torch.load(f'data/{args.dataset}_idx_train.pt')
idx_val = torch.load(f'data/{args.dataset}_idx_val.pt')
idx_test = torch.load(f'data/{args.dataset}_idx_test.pt')
#making features for cvae


# model definition
model = GIN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.num_classes,
    num_layers=args.num_layers,
).to(device)

model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# loading pretrain cvae model
cvae_model = torch.load("model/MUTAG.pkl") # load

def get_augmented_features(concat):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features)
        augmented_features = feature_tensor_normalize(augmented_features).detach()
        X_list.append(augmented_features.to(device))
    return X_list


# testing the model
@torch.no_grad()
def test_lagin(data, idx, concat):
    model.eval()
    xlist = get_augmented_features(concat) + [features_normalized]
    dataclones = [data.clone().to(device) for x in xlist]
    for i in range(len(xlist)):
        dataclones[i].x = xlist[i].to(device)
    out = model(dataclones)
    pred = out.argmax(dim=-1)

    total_correct = int((pred[idx] == data.y[idx]).sum())
    return total_correct / len(idx)


# main training loop

test_res = []
accs = []
for run in range(1,args.num_runs+1):
    model = LAGIN(
        concat=args.concat+1,
        nfeat=dataset.num_features,
        nhid=args.hidden_channels,
        out_ch=dataset.num_classes,
        num_layers=7,
    ).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5, amsgrad=True)

    best_val_acc = final_test_acc = 0
    pbar = tqdm(range(1, args.n_epochs+1), ncols=850, leave=False)
    for epoch in pbar:
        pbar.set_description_str(f"(Run {run}) Epoch: {epoch:3d}|{args.n_epochs}")
        model.train()
        output_list = []
        for k in range(int(args.samples)):
            X_list = get_augmented_features(args.concat)
            X_list = X_list + [features_normalized]
            dataclones = [big_graph.clone().to(device) for x in X_list]
            for i in range(len(X_list)):
                dataclones[i].x = X_list[i].to(device)
            output_list.append(model(dataclones))

        loss = 0
        for n, output in enumerate(output_list):
            l = F.cross_entropy(output[idx_train].to(device), big_graph.y[idx_train].to(device))
            loss += l/len(output_list)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

        val_acc = test_lagin(big_graph.to(device), idx_val.to(device), torch.tensor(args.concat).to(device))
        test_acc = test_lagin(big_graph.to(device), idx_test.to(device), torch.tensor(args.concat).to(device))

        accs.append([val_acc, test_acc])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
        pbar.set_postfix_str(f'loss: {loss:.4f}, val: {val_acc:.4f}, '
            f'test: {test_acc:.4f}')
    test_res.append(final_test_acc)

print(f"test_mu: {np.mean(test_res):.4f}, test_std: {np.std(test_res):.4f} (over {args.num_runs} runs)")
