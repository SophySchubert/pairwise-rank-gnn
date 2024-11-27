import networkx as nx
from node2vec import Node2Vec
import plotly.graph_objects as go
import time
import pickle
import matplotlib.pyplot as plt


def vis():
    '''
    Visualize a network graph from https://plotly.com/python/network-graphs/
    '''
    G = nx.random_geometric_graph(200, 0.125)
    print(f"Nodes:{G.nodes()}")
    print(f"Edges:{G.edges()}")
    #################################################################################
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
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

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Node Connections',
                    side='right'
                ),
                xanchor='left',
            ),
            line_width=2))
    #################################################################################
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: ' + str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    #################################################################################
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text="<br>Network graph made with Python",
                            font=dict(
                                size=16
                            )
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Python code: <a href='https://plotly.com/python/network-graphs/'> https://plotly.com/python/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()

def simple_vis(graph):
    # Define size
    num_nodes = len(graph.nodes())
    base_size = 1000
    node_size = base_size / num_nodes

    # Get positions
    pos = None
    if nx.get_node_attributes(graph, 'pos') == {}:
        pos = nx.spring_layout(graph)  # positions for all nodes
    else:
        pos = nx.get_node_attributes(graph, 'pos')

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=node_size)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), width=node_size * 0.01)

    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=node_size * 0.05)

    # Display the graph
    plt.show()

def graph2embedding():
    # TODO: expand to work with real dataset
    EMBEDDING_FILENAME = "temp.txt"
    EMBEDDING_MODEL_FILENAME = "temp.emb"

    graph = nx.fast_gnp_random_graph(n=100, p=0.5)

    # print(graph.nodes())
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs
    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Look for most similar nodes
    model.wv.most_similar('2')  # Output node names are always strings

    # Save embeddings for later use
    model.wv.save_word2vec_format(EMBEDDING_FILENAME)

    # Save model for later use
    # model.save(EMBEDDING_MODEL_FILENAME)

def saveGraph(graph):
    with open("temp.p", "wb") as f:
        pickle.dump(graph, f)

def loadGraph(name:'temp.p'):
    with open("temp.p", "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    start_time = time.time()

    G = nx.Graph()
    G.add_node("A")
    G.add_node("B")
    G.add_node("C")
    G.add_node("D")
    G.add_edge("A", "B", weight=4)
    G.add_edge("B", "D", weight=2)
    G.add_edge("A", "C", weight=3)
    G.add_edge("C", "D", weight=4)
    simple_vis(G)

    print("--- %s seconds ---" % (time.time() - start_time))