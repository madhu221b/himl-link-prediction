import networkx as nx
import numpy as np
import pandas as pd
from fairwalk.fairwalk  import FairWalk
from utils import set_seed, rewiring_list, recommender_model, get_top_recos


# Hyperparameter for FairWalk
DIM = 64
WALK_LEN = 10
NUM_WALKS = 200


def get_edges_from_annotator(g,u,seed):
    """


    """
    set_seed(seed)
    print("Generating Embeddings from Fairwalk Model")
    fairwalk_model, embeddings = recommender_model(g, model="fw")

    print("Getting Link Recommendations from Annotator")
    recos = get_top_recos(g,embeddings, u) 
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