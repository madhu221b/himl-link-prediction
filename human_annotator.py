import networkx as nx
import numpy as np

from fairwalk.fairwalk  import FairWalk
from utils import set_seed

# Hyperparameter for FairWalk
DIM = 64
WALK_LEN = 10
NUM_WALKS = 200

def recommender_model(G, num_cores=8):
    fw_model = FairWalk(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores)
    model = fw_model.fit() 
    return model

def get_top_recos(model, u):
    results = []
    for ns in u:
        nt = int(model.wv.most_similar(str(ns))[0][0]) # generate top 1 recommendation
        results.append((ns,nt))
    return results

def rewiring_list(G, node, number_of_rewiring):
        nodes_to_be_unfollowed = []
        node_neighbors = np.array(list(G.successors(node)))
        nodes_to_be_unfollowed = np.random.permutation(node_neighbors)[:number_of_rewiring]
        return list(map(lambda x: tuple([node, x]), nodes_to_be_unfollowed))

def get_edges_from_annotator(g,u,seed):
    """


    """
    set_seed(seed)
    print("Generating Embeddings from Fairwalk Model")
    fairwalk_model = recommender_model(g)

    print("Getting Link Recommendations from Annotator")
    recos = get_top_recos(fairwalk_model, u) 

    for i,(u,v) in enumerate(recos):
         seed += i
         set_seed(seed)
         edges_to_be_removed = rewiring_list(g, u, 1)
         g.remove_edges_from(edges_to_be_removed) # deleting previously existing links
         g.add_edge(u,v)
    return g