import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
from matplotlib.patches import Circle
from community import community_louvain
random.seed(123)
np.random.seed(123)
"""
 # to install networkx 2.0 compatible version of python-louvain use:
 # pip install -U git+https://github.com/taynaud/python-louvain.git@networkx2
"""
# g_path = "/home/mpawar/himl-link-prediction/_human/seed_42/B_75/dim_64/DPAH/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.2-hmm0.8-ID0.gpickle_n_epoch_9.gpickle"
# g_path = "/home/mpawar/himl-link-prediction/_no_human/seed_42/B_0/dim_64/DPAH/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.2-hmm0.8-ID0.gpickle_n_epoch_9.gpickle"


def get_hoG(partition,g):
    comm_dict = {} # community_no: {"no of minority nodes:"{}, "total no of nodes:"}
    for node, community in partition.items():
        c = g.nodes[node]["m"]
        if community not in comm_dict:
            comm_dict[community] = {"m":0, "t":0}

        comm_dict[community]["t"] += 1
        if c == 1:
            comm_dict[community]["m"] += 1

    avg_dict = {no:np.round((val_dict["m"]/val_dict["t"])*100.0,2) for no, val_dict in comm_dict.items()}
    return avg_dict.values()

def visualize_hoG(avg_vals,fm,hmm,hMM,t,no_human):
    num_bins = 10
    fig, ax = plt.subplots() 
    if no_human: 
        model = "no_human"
        color = "#DD654B"
    else:
        model = "human"
        color = "#81B622"
    n, bins, patches = ax.hist(avg_vals, num_bins, 
                           density = 1,  
                           color =color,  
                           alpha = 0.7) 
    ax.set_ylim(0,0.2)
    file_name = "plots/hoG_{}_fm{}_hMM{}_hmm{}_nepoch{}.png".format(model,fm,hMM,hmm,t)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')

def run(fm,hmm,hMM,no_human):

    T = [0,3,6,9]
    for t in T:
        if no_human:
          g_path = "../himl-link-prediction/_no_human/seed_42/B_0/dim_64/DPAH/DPAH-N1000-fm{}-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}-ID0.gpickle_n_epoch_{}.gpickle".format(fm,hMM,hmm,t)
        else:
          g_path = "../himl-link-prediction/_human/seed_42/B_75/dim_64/DPAH/DPAH-N1000-fm{}-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}-ID0.gpickle_n_epoch_{}.gpickle".format(fm,hMM,hmm,t)

        g = nx.read_gpickle(g_path)
        partition = community_louvain.best_partition(g.to_undirected(),resolution=1.2)
        avg_vals = get_hoG(partition,g)
        visualize_hoG(avg_vals,fm,hmm,hMM,t,no_human)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hMM", help="homophily between Majorities", type=float, default=0.5)
    parser.add_argument("--hmm", help="homophily between minorities", type=float, default=0.5)
    parser.add_argument("--fm", help="Minority Fraction", type=float, default=0.3)
    parser.add_argument("--no_human", default=False, action="store_true", help="Generate results without human intervention")
    args = parser.parse_args()
    run(args.fm,args.hmm,args.hMM,args.no_human)
   