import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
import scipy.sparse as sp
import torch_geometric.transforms as T
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv

dataset = PygNodePropPredDataset('ogbn-products')
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]

adj_scipy = sp.csr_matrix(PygNodePropPredDataset(name='ogbn-products', transform=T.ToSparseTensor())[0].adj_t.to_scipy())

train_idx = split_idx['train']
train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[15, 10, 5], batch_size=1024,
                               shuffle=True, num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  num_workers=12)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


parser = argparse.ArgumentParser(description='OGBN-Products (GNN)')

parser.add_argument("--concat", type=int, default=1)
parser.add_argument("--samples", type=int, default=4)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--hidden_channels', type=int, default=256)

parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')

args = parser.parse_args()
print(args)

def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, args.hidden_channels, dataset.num_classes, num_layers=3)
model = model.to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)


cvae_model = torch.load("model/products.pkl")
def get_augmented_features(concat):
    X_list = []
    for _ in range(concat):
        z = torch.randn([x.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, x).detach()
        X_list.append(augmented_features)
    return X_list


def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        output_list = []
        for k in range(args.samples):
            Augmented_X = get_augmented_features(args.concat)[0]
            output_list.append(model((Augmented_X[n_id]+x[n_id])/2, adjs).log_softmax(dim=-1))
        
        loss = 0.
        for k in range(args.samples):    
            loss += F.nll_loss(output_list[k], y[n_id[:batch_size]])
        
        loss = loss/len(output_list)
        loss_consis = consis_loss(output_list)
        loss = loss + loss_consis

        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(output_list[0].argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()
    val_Augmented_X = get_augmented_features(args.concat)[0]
    out = model.inference((val_Augmented_X+x)/2)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc

import logging
logging.basicConfig(filename='output.log',
                    filemode='a',
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

ilogger = logging.getLogger(__file__.split("_")[-1][:-3])
ilogger.info(f"Logging results for {__file__}")

test_accs = []
for run in range(1, 11):
    # print('')
    ilogger.info(f'Run {run:02d}:')
    # print('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val_acc = final_test_acc = 0
    for epoch in range(1, 21):
        loss, acc = train(epoch)
        ilogger.info(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

        if epoch > 5:
            train_acc, val_acc, test_acc = test()
            ilogger.info(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    test_accs.append(final_test_acc)

test_acc = torch.tensor(test_accs)
ilogger.info('============================')
ilogger.info(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
