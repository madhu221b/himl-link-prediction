import argparse
import networkx as nx
from tqdm import tqdm
import pickle as pkl
from utils import get_filename, run_acquisition_function, set_seed, save_metadata
from human_annotator import get_edges_from_annotator
from generate_results import get_visibility_plot, get_centrality_plot, time_vs_betn, time_vs_visibility
from dist_based_recos import train
MAIN_SEED = 42

def make_one_timestep(g, seed, dim=64,B=10):
    
    # Step 2: Get source node u from Acquisition Function
    print("!! [Step 2]  Picking Samples from Acquisition Function !!")
    u = run_acquisition_function(g,B)

    # Step 3: Get Edge(s) from Annotator
    print("!! [Step 3]  Generating Labels from Human Annotator !!")
    g_new = get_edges_from_annotator(g.copy(),u,seed)

    # Step 4: Train Model
    print("!! [Step 4]  Train Model !!")
    g_changed = train(g_new, seed, dim)

    return g_changed

def make_one_timestep_no_human(g, seed):
    
    print("!! [Step 1]  Train Model !!")
    g_new = train(g, seed)
    return g_new

def run(hMM, hmm,dim,T,B,seed):
    acc_dict = {}
    set_seed(seed)
    
    # Step 1: Read Initial Graph
    print("!! [Step 1] Get Graph of hMM={}, hmm={} !!".format(hMM,hmm))
    g = nx.read_gpickle(get_filename(hMM,hmm))
    node2group = {node:g.nodes[node]["m"] for node in g.nodes()}
    nx.set_node_attributes(g, node2group, 'group')

    iterable = tqdm(range(T), desc='Timesteps', leave=True) 
    time = 0
    seed_orig = seed
    for time in iterable:
        # seed = MAIN_SEED+time+1 
        seed = seed+time+1
        save_metadata(g,hMM,hmm,"_human",time,B,dim,seed_orig)
        g_test = make_one_timestep(g.copy(), dim, seed,B)
        g = g_test
 
 
def run_no_human(hMM, hmm,T,seed):
    set_seed(seed)
    acc_dict = {}
    
    # Step 1: Read Initial Graph
    print("!! [Step 1] Get Graph of hMM={}, hmm={} !!".format(hMM,hmm))
    g = nx.read_gpickle(get_filename(hMM,hmm))


    iterable = tqdm(range(T), desc='Timesteps', leave=True) 
    time = 0
    seed_orig = seed
    for time in iterable:
        seed = seed+time+1 

        save_metadata(g,hMM,hmm,"_no_human",time,seed=seed_orig)
        g_updated = make_one_timestep_no_human(g.copy(), seed)
        g = g_updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hMM", help="homophily between Majorities", type=float, default=0.5)
    parser.add_argument("--hmm", help="homophily between minorities", type=float, default=0.5)
    parser.add_argument("--T", help="Timesteps to continue the loop", type=int, default=10)
    parser.add_argument("--B", help="Active Learning Batch Size", type=int, default=75)
    parser.add_argument("--dim", help="Dimensionality of N2V", type=int, default=64)
    parser.add_argument("--seed", help="Seed", type=int, default=42)
    parser.add_argument("--evaluate", default=False, action="store_true", help="Generate results on generated test graphs")
    parser.add_argument("--no_human", default=False, action="store_true", help="Generate results without human intervention")
    args = parser.parse_args()
    
    if args.evaluate:
        # time_vs_betn(args.hMM, args.hmm)
        time_vs_visibility(args.hMM, args.hmm)
        # get_visibility_plot(args.hmm, args.hMM, args.B, args.no_human)
        # get_centrality_plot(args.hmm, args.hMM, args.B, args.no_human)
    else:
        if not args.no_human:
            for seed in [42,420,4200]:
                for B in [50,100,200]:
                   run(args.hMM, args.hmm, args.dim, args.T,B,seed)
        else:
            for seed in [42,420,4200]:
                # run_no_human(args.hMM, args.hmm, args.T,args.seed)
                print("Running for seed: ", seed)
                run_no_human(args.hMM, args.hmm, args.T,seed)
