import numpy as np
from ogb.graphproppred import GraphPropPredDataset
from spektral.data import Dataset, Graph


class OGBDataset(Dataset):
    '''
    (spektral) Dataset class wrapper for Open Graph Benchmark datasets.
    '''
    def __init__(self, name, **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def read(self):
        dataset = GraphPropPredDataset(name=self.name)
        graphs = []
        for data in dataset:
            edge_index = data[0]['edge_index']
            edge_feat = data[0]['edge_feat']
            node_feat = data[0]['node_feat']
            label = data[1]

            # Create adjacency matrix
            num_nodes = node_feat.shape[0]
            adj = np.zeros((num_nodes, num_nodes))
            for edge in edge_index.T:
                adj[edge[0], edge[1]] = 1

            # Create spektral Graph object
            graph = Graph(x=node_feat, a=adj, e=edge_feat, y=label)
            graphs.append(graph)

        return graphs

def ogb_available_datasets():
    #Datasets have size % 2 == 0 number of graphs
    return ['ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo']