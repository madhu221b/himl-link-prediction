import os
import networkx as nx
import numpy as np
import operator
import random

# constant paramaters of the graphs
N = 1000
fm = 0.3
d = 0.03
YM, Ym = 2.5, 2.5
model = "DPAH"

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def get_filename(hMM,hmm):
    full_path = "./DPAH"
    return os.path.join(full_path, "{}-N{}-fm{}{}{}{}{}{}-ID0.gpickle".format(model, N, 
                                             round(fm,1), 
                                             '-d{}'.format(round(d,5)), 
                                             '-ploM{}'.format(round(YM,1)), 
                                             '-plom{}'.format(round(Ym,1)), 
                                             '-hMM{}'.format(hMM),
                                             '-hmm{}'.format(hmm)))


def run_acquisition_function(g, B=1):
    """
    B : Size of Batch Active Learning
    """
    centrality_dict = nx.betweenness_centrality(g)
    minority_centrality = {node:val for node, val in centrality_dict.items() if g.nodes[node]["m"] == 1}
    sorted_dict = sorted(minority_centrality.items(), key=operator.itemgetter(1))[:B]
    u = [node for node, _ in sorted_dict]
    return u