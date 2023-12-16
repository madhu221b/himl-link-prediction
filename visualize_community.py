import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
from matplotlib.patches import Circle

random.seed(123)
np.random.seed(123)

g_path = "/home/mpawar/himl-link-prediction/_human/B_75/dim_64/DPAH/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.8-hmm0.2-ID0.gpickle_n_epoch_9.gpickle"

# colors = iter([plt.cm.tab20(i) for i in range(50)])
cmap = {name:plt.get_cmap(name) for name in ('Pastel1','Pastel2')}
N = 30
colors = np.concatenate([cmap[name](np.linspace(0, 1, N)) 
                         for name in ('Pastel2','Pastel1')])  
                         
color_dict = {'min':'#ec8b67', 'maj':'#6aa8cb'}
def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

   

    pos_nodes = _position_nodes(g, partition, scale=1.)
    pos_communities, circle_dict = _position_communities(g, partition, pos_nodes, scale=3.)
    # combine positions
    print("no of comms", len(circle_dict))
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos, circle_dict

def _position_communities(g, partition,pos_nodes, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    circle_dict = dict()
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    for community_no, coords in pos_communities.items():
        if community_no not in circle_dict:
            circle_dict[community_no] = dict()
        circle_dict[community_no]["coords"] = coords
        circle_dict[community_no]["r"] = 0

    # set node positions to position of community
    pos = dict()
    
   
    for node, community in partition.items():
        pos[node] = pos_communities[community]
        actual_node_pos = pos[node]
        total_node_pos = actual_node_pos + pos_communities[community]
        r = np.sqrt((pos_nodes[node][0])**2 + \
           (pos_nodes[node][1])**2)
        if r > circle_dict[community]["r"]:
            circle_dict[community]["r"] = r


    return pos, circle_dict

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def test():
    # to install networkx 2.0 compatible version of python-louvain use:
    # pip install -U git+https://github.com/taynaud/python-louvain.git@networkx2
    from community import community_louvain
    patches = [] 
  
  
    circle = Circle((-0.18039811, -0.17300569), 2,alpha=0.2) 

    # g = nx.karate_club_graph()
    g = nx.read_gpickle(g_path)
   
    partition = community_louvain.best_partition(g.to_undirected(),resolution=1.2)
 
    pos, circle_dict = community_layout(g, partition)

  
    fig, ax = plt.subplots() 
    node_color = [color_dict['min'] if obj['m'] else color_dict['maj'] for n,obj in g.nodes(data=True)]
    # nx.draw_networkx_edges(g,pos,alpha=0.2)
    
    # nx.draw(g, pos,node_color=node_color)
    nx.draw_networkx_nodes(g,pos,node_color=node_color,edgecolors="black",linewidths=0.2)
    
    for community_no, value in circle_dict.items():
        # [next(colors)][0
        circle = Circle(value["coords"],value["r"],color=colors[community_no],alpha=0.5,linewidth=2)
        ax.add_patch(circle)
    # ax.add_patch(circle)
  
    plt.savefig('plotgraph.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    test()
