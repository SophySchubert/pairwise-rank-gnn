from itertools import combinations
import numpy as np
from scipy.stats import rankdata, kendalltau
import networkx as nx
import torch
from torch_geometric.data.data import Data
from torch_geometric.utils.convert import from_networkx


def sample_preference_pairs(graphs):
    c = [(a, b, check_util(graphs, a,b)) for a, b in combinations(range(len(graphs)), 2)]
    return np.array(c)

def check_util(data, index_a, index_b):
    a = data[index_a]
    b = data[index_b]
    util_a = a.y
    util_b = b.y
    if util_a >= util_b:
        return 1
    else:
        return 0

def rank_data(items):
    return rankdata(items, method='dense')

def compare_rankings_with_kendalltau(ranking_a, ranking_b):
    return kendalltau(ranking_a, ranking_b)

def train(model, loader, device, optimizer, criterion):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        out = out[0]
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device, criterion):
    model.eval()
    error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            out = out[0]
            out = out.float()
            error += criterion(out, data.y.float())
    return error / len(loader)

def predict(model, loader, device):
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            utils = out[1].detach().cpu().numpy()
    return utils

def combine_two_graphs(graph_a, graph_b):
    # TODO: a) Kanten Beschriften, ob "standard" oder "Kombiniation"      [__]
    # TODO: b) Welchen Wert bekommt der Graph                                       [✓]
    ''''
    Combines two Graphs into one
    Visualisation in Eigenanteil.ipynb
    '''
    temp_graph = nx.full_join(graph_a, graph_b)
    temp_y = 1 if graph_a.y.item() >= graph_b.y.item() else 0
    graph = from_networkx(temp_graph)
    graph.y = temp_y
    return graph



####### DEPRECATED #######
def iterate_train_random(elements):
    pass
    # objects = elements
    # utilities = np.array([e.y.item() for e in elements])  # Convert tensor values to a numpy array
    # sort_idx = np.argsort(utilities, axis=0)
    # olen = len(objects)
    # seed = SEED + olen
    # pair_count = (olen * (olen - 1)) // 2
    # sampling_ratio = 1
    # sample_size = min(int(sampling_ratio * pair_count), pair_count)
    # rng = np.random.default_rng(seed)
    #
    # sample = rng.choice(pair_count, sample_size, replace=False)
    # sample_b = (np.sqrt(sample * 2 + 1 / 4) + 1 / 2).astype(int)  # Convert to integer type
    # sample_a = (sample - (sample_b * (sample_b - 1)) // 2).astype(int)  # Convert to integer type
    # idx_a = sort_idx[sample_a]
    # idx_b = sort_idx[sample_b]
    #
    # return idx_a, idx_b

def get_target(data, indices_a, indices_b):
    pass
    # assert len(indices_a) == len(indices_b)
    # util_a = np.array([data[idx].y.item() for idx in indices_a])
    # util_b = np.array([data[idx].y.item() for idx in indices_b])
    # target = (util_a > util_b).astype(int)
    # return target

if __name__ == "__main__":
    # Example usage for iterate_train_random
    # train_idx_a, train_idx_b = iterate_train_random(train_dataset)
    # # target = get_target(train_dataset, train_idx_a, train_idx_b)
    # train_target = np.zeros(len(train_idx_a))#due to argsort in iterate_train_randomB all targets are 0
    # train_prefs = np.array(list(zip(train_idx_a, train_idx_b, train_target)))

    pass