from itertools import combinations
import numpy as np
from scipy.stats import rankdata, kendalltau
import networkx as nx
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import to_dense_batch, to_dense_adj
from tqdm import tqdm

def sample_preference_pairs(graphs):
    '''
    Create each possible pair without repetition. At the same time create the target value
    '''
    c = [(a, b, check_util(graphs, a,b)) for a, b in combinations(range(len(graphs)), 2)]
    return np.array(c)

def check_util(data, index_a, index_b):
    '''
    Compare the utilites of both graphs and return the target value
    Target value is 1 if the first graph has a higher utility than the second graph.
    1 indicates that the first graph is preferred over the second graph.
    '''
    a = data[index_a]
    b = data[index_b]
    util_a = a.y
    util_b = b.y
    if util_a >= util_b:
        return 1
    else:
        return 0

def rank_data(items):
    '''
    Wrapper for rankdata from scipy.stats, with method='dense'
    Takes a list of items and returns their ranks - kind of like sorting but returns just the ranks
    '''
    return rankdata(items, method='dense')

def compare_rankings_with_kendalltau(ranking_a, ranking_b):
    '''
    Wrapper for kendalltau from scipy.stats
    Takes two lists and compares how similar they are
    '''
    return kendalltau(ranking_a, ranking_b)

def train(model, loader, device, optimizer, criterion, mode='default'):
    # Train-method for the model
    model.train()
    for data in tqdm(loader):
        if mode == 'default' or mode == 'fc_extra' or mode == 'fc' or mode == 'rank_mask':
            data = data.to(device)
            y = data.y
        elif mode == 'nagsl_attention':
            data = data #already on device
            y = data['target']
        else:#gat_attention
            tmp_0 = data[0].to(device)
            tmp_1 = data[1].to(device)
            data = [tmp_0, tmp_1]
            y = data[0].y
        optimizer.zero_grad()
        if mode == 'default':
            pref, util = model(data)
        else:
            pref = model(data)
            pref = pref.squeeze()
        loss = criterion(pref, y.float())
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device, criterion, mode='default'):
    # Evaluate-method for the model
    model.eval()
    error = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            if mode == 'default' or mode == 'fc_extra' or mode == 'fc' or mode == 'rank_mask':
                data = data.to(device)
                y = data.y
            elif mode == 'nagsl_attention':
                data = data #already on device
                y = data['target']
            else:#gat_attention
                tmp_0 = data[0].to(device)
                tmp_1 = data[1].to(device)
                data = [tmp_0, tmp_1]
                y = data[0].y

            if mode == 'default':
                pref, util = model(data)
            else:
                pref = model(data).squeeze()
            loss = criterion(pref, y.float())
            error += loss.item()

            #calc accuracy
            predicted = (pref >= 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total

    return error / len(loader), accuracy

def predict(model, loader, device, mode='default'):
    # Prediction-method for the model
    model.eval()
    with torch.no_grad():
        for data in loader:
            if mode == 'default' or mode == 'fc_extra' or mode == 'fc' or mode == 'rank_mask':
                data = data.to(device)
            elif mode == 'nagsl_attention':
                data = data  # already on device
            else:#gat_attention
                tmp_0 = data[0].to(device)
                tmp_1 = data[1].to(device)
                data = [tmp_0, tmp_1]

            if mode == 'default':
                pref, util = model(data)
                pref = (pref >= 0.5).float()
                pref = pref.detach().cpu().numpy()
                util = util.detach().cpu().numpy()
            else:#gat_attention, nagsl_attention, fc_extra, fc
                pref = model(data)
                pref = (pref >= 0.5).float()
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
    SLOW and DEPRECATED use combine_two_graphs_torch() instead
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
    '''
    Combines two Graphs via the Batch-object and adds fully connected edges between them
    Parameter: bidirectional - if True, adds edges in both directions
    '''
    # combine two graphs
    data_batch = Batch.from_data_list([graph_a, graph_b])
    num_nodes_graph_a = graph_a.num_nodes
    num_nodes_graph_b = graph_b.num_nodes
    node_features = data_batch.x
    edge_index = data_batch.edge_index
    edge_attr = data_batch.edge_attr
    target = 1 if graph_a.y.item() >= graph_b.y.item() else 0
    # fully connect two graphs
    adj = torch.cartesian_prod(torch.arange(num_nodes_graph_a),
                               torch.arange(num_nodes_graph_a, num_nodes_graph_a + num_nodes_graph_b))
    adj = adj.transpose(0, 1)  # have it look like edge_index
    if bidirectional:# also add the reverse edges
        adj0 = adj[0]
        adj1 = adj[1]
        adj0, adj1 = torch.cat((adj0, adj1)), torch.cat((adj1, adj0))
        adj = torch.stack((adj0, adj1))
    edge_index = torch.cat((edge_index, adj), axis=1)
    # create new graph
    new_graph = Data(x=node_features, edge_attr=edge_attr, edge_index=edge_index, y=target,
                     num_nodes=num_nodes_graph_a+num_nodes_graph_b)

    return new_graph

def transform_dataset_to_pair_dataset(dataset, prefs, config):
    '''
    SLOW and DEPRECATED use transform_dataset_to_pair_dataset_torch() instead
    Transforms every pair from prefs into a new combined graph
    '''
    new_dataset = []
    for pref in prefs:
        g_1, g_2 = dataset[pref[0]], dataset[pref[1]]
        combined_graph = combine_two_graphs(graph_a=g_1, graph_b=g_2, default_value=config['new_grap_edge_value'])
        assert(combined_graph.y == pref[2])
        new_dataset.append(combined_graph)
    return new_dataset

def transform_dataset_to_pair_dataset_torch(dataset, prefs, config, from_loader=False):
    '''
    Transforms every pair from prefs into a new combined graph
    '''
    new_dataset = []
    if from_loader: # needed for mode: rank_mask -> removes the possibility of incorrectly ordered graphs
        _d = []
    for pref in prefs:
        g_1, g_2 = dataset[pref[0]], dataset[pref[1]]
        combined_graph = combine_two_graphs_torch(graph_a=g_1, graph_b=g_2, bidirectional=config['bidirectional'])
        if from_loader:
            _d.append(g_1)
            _d.append(g_2)
        assert(combined_graph.y == pref[2])
        new_dataset.append(combined_graph)
    if from_loader:
        return new_dataset, _d
    return new_dataset

def preprocess_predictions(raw_predictions):
    '''
    Preprocesses the raw predictions from the model by switching the first and second indices if the last index is 0.
    Thus insuring that the first index is always the one with the higher utility and should be the preferred one.
    '''
    # Create a mask for rows where the last element is 0
    mask = raw_predictions[:, 2] == 0
    # Switch the first and second indices for rows where the mask is True
    raw_predictions[mask] = raw_predictions[mask][:, [1, 0, 2]]
    # Remove the last index
    cleaned_predictions = raw_predictions[:, :2]
    return cleaned_predictions

def retrieve_preference_counts_from_predictions(predictions, max_range):
    '''
    Takes the predictions and counts the number of times each element appears in the first index.
    '''
    # Extract the first column
    first_index_elements = predictions[:, 0]

    # Create a range of possible numbers from 0 to max_range
    possible_numbers = range(max_range)

    # Initialize the dictionary with all possible numbers
    element_counts = {num: 0 for num in possible_numbers}

    # Count the occurrences of each element in the first index
    unique_elements, counts = np.unique(first_index_elements, return_counts=True)

    # Update the dictionary with the actual counts
    element_counts.update(dict(zip(unique_elements, counts)))

    return list(element_counts.values())

def nagsl_pair_attention_transform(data, target, config):
    '''
    Method to transform the data for the NAGSL model
    '''
    new_data = dict()

    b0 = to_dense_batch(data[0].x, batch=data[0].batch, max_num_nodes=config['max_num_nodes'])
    g0 = {
        'adj': to_dense_adj(
            data[0].edge_index, batch=data[0].batch, max_num_nodes=config['max_num_nodes']
        ).to(config['device']),
        'x': b0[0].to(config['device']),
        'mask': b0[1].to(config['device']),
        'dist': None
    }

    b1 = to_dense_batch(data[1].x, batch=data[1].batch, max_num_nodes=config['max_num_nodes'])
    g1 = {
        'adj': to_dense_adj(
            data[1].edge_index, batch=data[1].batch, max_num_nodes=config['max_num_nodes']
        ).to(config['device']),
        'x': b1[0].to(config['device']),
        'mask': b1[1].to(config['device']),
        'dist': None
    }

    new_data['g0'] = g0
    new_data['g1'] = g1
    new_data['target'] = target.to(config['device'])
    return new_data