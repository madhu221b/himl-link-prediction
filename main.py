import argparse
import networkx as nx
from tqdm import tqdm
from utils import get_filename, run_acquisition_function, set_seed
from human_annotator import get_edges_from_annotator
from model import train
MAIN_SEED = 42

def make_one_timestep(g, seed):
    
    # Step 2: Get source node u from Acquisition Function
    print("!! [Step 2]  Picking Samples from Acquisition Function !!")
    u = run_acquisition_function(g,B=3)

    # Step 3: Get Edge(s) from Annotator
    print("!! [Step 3]  Generating Labels from Human Annotator !!")
    g_new = get_edges_from_annotator(g.copy(),u,seed)

    # Step 4: Train Model
    print("!! [Step 4]  Train Model !!")
    train(g_new, seed)

    return g_new
  
def run(hMM, hmm,T):
    set_seed(MAIN_SEED)
    
    # Step 1: Read Initial Graph
    print("!! [Step 1] Get Graph of hMM={}, hmm={} !!".format(hMM,hmm))
    g = nx.read_gpickle(get_filename(hMM,hmm))
    node2group = {node:g.nodes[node]["m"] for node in g.nodes()}
    nx.set_node_attributes(g, node2group, 'group')

    iterable = tqdm(range(T), desc='Timesteps', leave=True) 
    time = 0
    for time in iterable:
        seed = MAIN_SEED+time+1 
        g_updated = make_one_timestep(g.copy(), seed)
        g = g_updated
        
 

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hMM", help="homophily between Majorities", type=float, default=0.5)
    parser.add_argument("--hmm", help="homophily between minorities", type=float, default=0.5)
    parser.add_argument("--T", help="Timesteps to continue the loop", type=int, default=10)
    parser.add_argument("--evaluate", default=False, action="store_true", help="Generate results on generated test graphs")
    args = parser.parse_args()
    

    run(args.hMM, args.hmm, args.T)
