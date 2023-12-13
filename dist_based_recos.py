import numpy as np
import networkx as nx
from utils import set_seed, rewiring_list, recommender_model, get_top_recos



def train(g, seed,dim=64):

    print("Generating Node Embeddings")
    n2v_model, n2v_embeds = recommender_model(g,n2v_dim=dim,model="n2v")
    print("Getting Link Recommendations from N2V Model")
    u = g.nodes()
    recos = get_top_recos(g,n2v_embeds, u) 
    no_new_edges = 0
    for i,(u,v) in enumerate(recos):
         seed += i
         set_seed(seed)
         if not g.has_edge(u,v):
            no_new_edges += 1
            edges_to_be_removed = rewiring_list(g, u, 1)
            g.remove_edges_from(edges_to_be_removed) # deleting previously existing links
            g.add_edge(u,v)
    return g









     