from itertools import combinations
import numpy as np
from scipy.stats import rankdata, kendalltau
import networkx as nx
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import to_networkx, from_networkx

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

def train(model, loader, device, optimizer, criterion, mode='default'):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        if mode == 'default':
            pref, util = model(data)
        else:
            pref = model(data).squeeze()
        loss = criterion(pref, data.y.float())
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device, criterion, mode='default'):
    model.eval()
    error = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if mode == 'default':
                pref, util = model(data)
            else:
                pref = model(data).squeeze()
            loss = criterion(pref, data.y.float())
            error += loss.item()

            #calc accuracy
            predicted = (pref >= 0.5).float()
            correct += (predicted == data.y).sum().item()
            total += data.y.size(0)

    accuracy = correct / total

    return error / len(loader), accuracy

def predict(model, loader, device, mode='default'):
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if mode == 'default':
                pref, util = model(data)
                pref = pref.detach().cpu().numpy()
                util = util.detach().cpu().numpy()
            else:
                pref = model(data)
                util = None
                pref = pref.detach().cpu().numpy()
    return pref, util

def convert_torch_to_nx(graph):
    '''
    Converts a torch_geometric graph to a networkx graph
    Includes nodes, edges, node features, and edge attributes
    :param graph: torch_geometric graph object
    :return: networkx graph object
    '''
    nx_g = to_networkx(graph)

    # Get edge attributes
    edge_attrs = graph.edge_attr
    # Create a dictionary of edge attributes
    edge_attr_dict = {}
    for i, (u, v) in enumerate(nx_g.edges()):
        edge_attr_dict[(u, v)] = edge_attrs[i].tolist()
    # Set edge attributes
    nx.set_edge_attributes(nx_g, edge_attr_dict, 'edge_attr')

    # Get node features
    node_features = graph.x
    # Create a dictionary of node features
    node_attr_dict = {}
    for i, node in enumerate(nx_g.nodes()):
        node_attr_dict[node] = node_features[i].tolist()
    # Set node features
    nx.set_node_attributes(nx_g, node_attr_dict, 'x')

    return nx_g

def combine_two_graphs(graph_a, graph_b, default_value=[1,1,1]):
    ''''
    Combines two Graphs into one with networx (kinda slow) sets a default value for edge weight
    Visualisation in Eigenanteil.ipynb
    '''
    # Convert graphs to networkx
    nx_graph_a = convert_torch_to_nx(graph_a)
    nx_graph_b = convert_torch_to_nx(graph_b)
    # Combine graphs
    temp_graph = nx.full_join(nx_graph_a, nx_graph_b, rename=("G", "H"))

    # Ensure all edges have the same attributes
    default_attrs = {'edge_attr': default_value}
    for u, v in temp_graph.edges():
        for attr, value in default_attrs.items():
            temp_graph[u][v].setdefault(attr, value)

    # calc y value
    temp_y = 1 if graph_a.y.item() >= graph_b.y.item() else 0

    graph = from_networkx(temp_graph)
    graph.y = temp_y
    graph.num_nodes = graph_a.num_nodes + graph_b.num_nodes

    return graph

def combine_two_graphs_torch(graph_a, graph_b, bidirectional=False):
    # combine two graphs
    data_batch = Batch.from_data_list([graph_a, graph_b])
    num_nodes_graph_a = graph_a.num_nodes
    num_nodes_graph_b = graph_b.num_nodes
    node_features = data_batch.x
    edge_index = data_batch.edge_index
    edge_attr = data_batch.edge_attr
    target = 1 if graph_a.y.item() >= graph_b.y.item() else 0
    # connect two graphss
    adj = torch.cartesian_prod(torch.arange(num_nodes_graph_a),
                               torch.arange(num_nodes_graph_a, num_nodes_graph_a + num_nodes_graph_b))
    adj = adj.transpose(0, 1)  # have it look like edge_index
    if bidirectional:
        adj0 = adj[0]
        adj1 = adj[1]
        adj0, adj1 = torch.cat((adj0, adj1)), torch.cat((adj1, adj0))
        adj = torch.stack((adj0, adj1))
    # create new graph
    new_graph = Data(x=node_features, edge_attr=edge_attr, edge_index=edge_index, y=target, adj=adj,
                     num_nodes=num_nodes_graph_a + num_nodes_graph_b)

    return new_graph

def transform_dataset_to_pair_dataset(dataset, prefs, config):
    '''
    Transforms every pair from prefs into a new combined graph
    '''
    new_dataset = []
    for pref in prefs:
        g_1, g_2 = dataset[pref[0]], dataset[pref[1]]
        combined_graph = combine_two_graphs(graph_a=g_1, graph_b=g_2, default_value=config['new_grap_edge_value'])
        assert(combined_graph.y == pref[2])
        new_dataset.append(combined_graph)
    return new_dataset

def transform_dataset_to_pair_dataset_torch(dataset, prefs, config):
    '''
    Transforms every pair from prefs into a new combined graph
    '''
    new_dataset = []
    for pref in prefs:
        g_1, g_2 = dataset[pref[0]], dataset[pref[1]]
        combined_graph = combine_two_graphs_torch(graph_a=g_1, graph_b=g_2, bidirectional=config['bidirectional'])
        assert(combined_graph.y == pref[2])
        new_dataset.append(combined_graph)
    return new_dataset

def preprocess_predictions(raw_predictions):
    # Create a mask for rows where the last element is 0
    mask = raw_predictions[:, 2] == 0
    # Switch the first and second indices for rows where the mask is True
    raw_predictions[mask] = raw_predictions[mask][:, [1, 0, 2]]
    # Remove the last index
    cleaned_predictions = raw_predictions[:, :2]
    return cleaned_predictions

def retrieve_preference_counts_from_predictions(predictions, max_range):
    # Extract the first column
    first_index_elements = predictions[:, 0]

    # Define the range of possible numbers (assuming 0 to 3 for this example)
    possible_numbers = range(max_range)

    # Initialize the dictionary with all possible numbers
    element_counts = {num: 0 for num in possible_numbers}

    # Count the occurrences of each element in the first index
    unique_elements, counts = np.unique(first_index_elements, return_counts=True)

    # Update the dictionary with the actual counts
    element_counts.update(dict(zip(unique_elements, counts)))

    return list(element_counts.values())


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