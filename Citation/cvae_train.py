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
from gcn.gin_models import GIN, LAGIN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
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

dataset = TUDataset("/content/datasets/", name=args.dataset).shuffle()
train_loader = gnnDataLoader(dataset[:0.9], args.batch_size, shuffle=True)
test_loader = gnnDataLoader(dataset[0.9:], args.batch_size)

big_graph = next(iter(gnnDataLoader(dataset, len(dataset))))
adj = to_scipy_sparse_matrix(big_graph.edge_index).tocsr()
features = big_graph.x
labels = big_graph.y
batch = big_graph.batch
idxs = torch.arange(len(dataset))
idxs = idxs[torch.randperm(len(idxs))]
idx_train, idx_val, idx_test = idxs[:int(len(idxs)*0.7)], idxs[int(len(idxs)*0.7):int(len(idxs)*0.85)], idxs[int(len(idxs)*0.85):]
features_normalized = feature_tensor_normalize(big_graph.x)
# will have to pass the idxs from here
torch.save(idx_train, f'feature/{args.dataset}_idx_train.pt')
torch.save(idx_val, f'feature/{args.dataset}_idx_val.pt')
torch.save(idx_test, f'feature/{args.dataset}_idx_test.pt')

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


t = 0
best_augmented_features  = None
cvae_model = None
best_score = -float("inf")

model = GIN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.num_classes,
    num_layers=args.num_layers,
).to(device)

model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

cvae = VAE(
    encoder_layer_sizes=[features.shape[1], 256],
    latent_size=args.latent_size,
    decoder_layer_sizes=[256, features.shape[1]],
    conditional=True,
    conditional_size=features.shape[1]
).to(device)
cvae_optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-2)

# small pretraining
epochs = 200

for _ in range(epochs // 2):
    output = model(big_graph.x.to(device), big_graph.edge_index.to(device), big_graph.batch.to(device))
    output = torch.log_softmax(output, dim=1)
    loss_train = F.nll_loss(output[idx_train].to(device), labels[idx_train].to(device))
    loss_train.backward()
    model_optimizer.step()

# cvae training loop

for _ in trange(args.pretrain_epochs, desc='Run CVAE Train', ncols=850):
    pbar = tqdm(cvae_dataset_dataloader, ncols=850)
    for _, (x, c) in enumerate(pbar):
        cvae.train()
        x, c = x.to(device), c.to(device)
        recon_x, mean, log_var, _ = cvae(x, c)
        cvae_loss = loss_fn(recon_x, x, mean, log_var)

        cvae_optimizer.zero_grad()
        cvae_loss.backward()
        cvae_optimizer.step()


        z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
        augmented_features = cvae.inference(z, cvae_features)
        augmented_features = feature_tensor_normalize(augmented_features).detach()

        big_graph_clone = big_graph.clone()
        big_graph_clone.x = augmented_features
        total_logits = 0
        cross_entropy = 0
        for i in range(args.num_models):
            logits = model(big_graph_clone.x.to(device), big_graph_clone.edge_index.to(device), big_graph_clone.batch.to(device))
            total_logits += F.softmax(logits, dim=1)
            output = F.log_softmax(logits, dim=1)
            cross_entropy += F.nll_loss(output[idx_train].to(device), labels[idx_train].to(device))
        output = torch.log(total_logits / args.num_models)
        U_score = F.nll_loss(output[idx_train].to(device), labels[idx_train].to(device)) - cross_entropy / args.num_models
        t += 1
        pbar.set_postfix_str(f"U Score: {U_score.item():.4f}, Best Score: {torch.tensor(best_score).item():.4f}")
        if U_score > best_score:
            best_score = U_score
            if t > args.warmup:
                cvae_model = copy.deepcopy(cvae)
                print("U_score: ", U_score, " t: ", t)
                best_augmented_features = copy.deepcopy(augmented_features)
                big_graph_clone = big_graph.clone()
                big_graph_clone.x = best_augmented_features

                for i in range(args.update_epochs):
                    model.train()
                    model_optimizer.zero_grad()
                    output = model(big_graph_clone.x.to(device), big_graph_clone.edge_index.to(device), big_graph_clone.batch.to(device))
                    output = torch.log_softmax(output, dim=1)
                    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                    loss_train.backward()
                    model_optimizer.step()


if cvae_model==None : cvae_model = copy.deepcopy(cvae)

torch.save(cvae_model, "model/MUTAG.pkl")
