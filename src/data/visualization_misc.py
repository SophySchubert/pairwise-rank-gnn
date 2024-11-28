import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx

def vis(graph):
    '''
    Visualize a graph from TUDataset with Plotly
    requires a networkx graph (convertion with spektral_to_networkx())
    '''

    pos = get_positions(graph)

    # nodes
    node_x = []
    node_y = []
    for key, val in pos.items():
        x, y = val
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='none', marker=dict(color='blue',
                                                                                              size=10))
    # edges
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        node1, node2 = edge
        x0, y0 = pos[node1]
        x1, y1 = pos[node2]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.show()

def simple_vis(graph):
    '''
    using matplotlib
    '''
    # Define size
    num_nodes = len(graph.nodes())
    base_size = 1000
    node_size = base_size / num_nodes

    pos = get_positions(graph)

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=node_size)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), width=node_size * 0.01)

    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=node_size * 0.05)

    # Display the graph
    plt.show()

def get_positions(graph):
    if nx.get_node_attributes(graph, 'pos') == {}:
        return nx.spring_layout(graph, seed=1)  # creates positions for all nodes
    else:
        return nx.get_node_attributes(graph, 'pos')