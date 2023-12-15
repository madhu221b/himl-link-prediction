import os
import networkx as nx
import numpy as np
import pandas as pd
import pickle as pkl
import operator
import random
from joblib import Parallel
from joblib import delayed
from collections import Counter
from fast_pagerank import pagerank_power
from sklearn.metrics.pairwise import cosine_similarity
from fairwalk.fairwalk  import FairWalk
from node2vec import Node2Vec

# constant paramaters of the graphs
N = 1000
fm = 0.3
d = 0.03
YM, Ym = 2.5, 2.5
model = "DPAH"
TOPK = 10

# Hyperparameter for node2vec/fairwalk
DIM = 64
WALK_LEN = 10
NUM_WALKS = 200


def recommender_model(G,n2v_dim=64,model="n2v",num_cores=8):
    if model == "n2v":
       node2vec = Node2Vec(G, dimensions=n2v_dim, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores)
       model = node2vec.fit() 
       emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    elif model == "fw":
        fw_model = FairWalk(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores)
        model = fw_model.fit() 
        emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    return model, emb_df


def get_top_recos(g, embeddings, u, N=1):
    all_nodes = g.nodes()
    df = embeddings
    results = []
    for src_node in u:
        source_emb = df[df.index == src_node]
        other_nodes = [n for n in all_nodes if n not in list(g.adj[src_node]) + [src_node]]
        other_embs = df[df.index.isin(other_nodes)]

        sim = cosine_similarity(source_emb, other_embs)[0].tolist()
        idx = other_embs.index.tolist()

        idx_sim = dict(zip(idx, sim))
        idx_sim = sorted(idx_sim.items(), key=lambda x: x[1], reverse=True)
        
        similar_nodes = idx_sim[:N]
        v = [tgt[0] for tgt in similar_nodes][0]
        results.append((src_node,v))
       
    return results 


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def create_subfolders(fn):
    path = os.path.dirname(fn)
    os.makedirs(path, exist_ok = True)

def save_gpickle(G, fn):
    try:
        create_subfolders(fn)
        nx.write_gpickle(G, fn)
        print('{} saved!'.format(fn))
    except Exception as ex:
        print(ex)

def save_csv(df, fn):
    try:
        create_subfolders(fn)
        df.to_csv(fn)
        print('{} saved!'.format(fn))
    except Exception as ex:
        print(ex)
        return False
    return True

def get_filename(hMM,hmm,fm=0.3):
    full_path = "./DPAH"
    return os.path.join(full_path, "{}-N{}-fm{}{}{}{}{}{}-ID0.gpickle".format(model, N, 
                                             round(fm,1), 
                                             '-d{}'.format(round(d,5)), 
                                             '-ploM{}'.format(round(YM,1)), 
                                             '-plom{}'.format(round(Ym,1)), 
                                             '-hMM{}'.format(hMM),
                                             '-hmm{}'.format(hmm)))


def get_fraction_in_topk(file_name,topk=10):
        df = pd.read_csv(file_name)
        
        # compute no of entries in topk
        total = len(df)
        k = round(topk/100.,2)    # k%
        t = int(round(k*total))   # No. of unique ranks in top-k

        df.sort_values(by='pagerank', ascending=False, inplace=True)
        topnodes = df[0:t] # first top-k ranks (list)
        fm_hat = topnodes.minority.sum()/topnodes.shape[0]
        return fm_hat

def get_centrality(file_name,centrality=""):
        g = nx.read_gpickle(file_name)

        if centrality == "betweenness":
            centrality_dict = nx.betweenness_centrality(g)
        elif centrality == "closeness":
            centrality_dict = nx.closeness_centrality(g)
        else:
            print("Invalid Centrality measure")
            return        
        
        minority_centrality = [val for node, val in centrality_dict.items() if g.nodes[node]["m"] == 1]
        avg_val = np.mean(minority_centrality)
        return avg_val

def run_acquisition_function(g, B=1):
    """
    B : Size of Batch Active Learning
    """
    centrality_dict = nx.betweenness_centrality(g)
    minority_centrality = {node:val for node, val in centrality_dict.items() if g.nodes[node]["m"] == 1}
    sorted_dict = sorted(minority_centrality.items(), key=operator.itemgetter(1))[:B]
    u = [node for node, _ in sorted_dict]
    return u


def rewiring_list(G, node, number_of_rewiring):
        nodes_to_be_unfollowed = []
        node_neighbors = np.array(list(G.successors(node)))
        nodes_to_be_unfollowed = np.random.permutation(node_neighbors)[:number_of_rewiring]
        return list(map(lambda x: tuple([node, x]), nodes_to_be_unfollowed))


def get_node_metadata_as_dataframe(g, njobs=1):
    cols = ['node','minority','indegree','outdegree','pagerank','wtf']
    df = pd.DataFrame(columns=cols)
    nodes = g.nodes()
    # minority = [g.node[n][g.graph['label']] for n in nodes]
    minority = [g.nodes[n][g.graph['label']] for n in nodes]
    indegree = [g.in_degree(n) for n in nodes]
    outdegree = [g.out_degree(n) for n in nodes]
    A = nx.to_scipy_sparse_matrix(g,nodes)
    # A = nx.attr_sparse_matrix(g)[0]
    pagerank = pagerank_power(A, p=0.85, tol=1e-6)
    wtf = who_to_follow_rank(A, njobs)
    
    return pd.DataFrame({'node':nodes,
                        'minority':minority,
                        'indegree':indegree,
                        'outdegree':outdegree,
                        'pagerank':pagerank,
                        'wtf':wtf,
                        }, columns=cols)

def save_metadata(g, hMM, hmm, model,n_epoch,fm,B=0,dim=64,seed=42):
    folder_path = "../himl-link-prediction/{}/seed_{}/B_{}/dim_{}".format(model,seed,B,dim)
    create_subfolders(folder_path)
    filename = get_filename(hMM, hmm,fm)
    
    fn = os.path.join(folder_path,'{}_n_epoch_{}.gpickle'.format(filename,n_epoch))
    save_gpickle(g, fn)

    ## [Personal] Specifying jobs
    njobs = 24
    df = get_node_metadata_as_dataframe(g, njobs=njobs)
    csv_fn = os.path.join(folder_path,'{}_n_epoch_{}.csv'.format(filename,n_epoch))
    save_csv(df, csv_fn)
    
    print("Saving graph and csv file at, ", filename)

def _ppr(node_index, A, p, top):
    pp = np.zeros(A.shape[0])
    pp[node_index] = A.shape[0]
    pr = pagerank_power(A, p=p, personalize=pp)
    pr = pr.argsort()[-top-1:][::-1]
    #time.sleep(0.01)
    return pr[pr!=node_index][:top]

def get_circle_of_trust_per_node(A, p=0.85, top=10, num_cores=40):
    return Parallel(n_jobs=num_cores)(delayed(_ppr)(node_index, A, p, top) for node_index in np.arange(A.shape[0]))

def frequency_by_circle_of_trust(A, cot_per_node=None, p=0.85, top=10, num_cores=40):
    results = cot_per_node if cot_per_node is not None else get_circle_of_trust_per_node(A, p, top, num_cores)
    unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
    del(results)
    return [ 0 if node_index not in unique_elements else counts_elements[np.argwhere(unique_elements == node_index)[0, 0]] for node_index in np.arange(A.shape[0])]

def _salsa(node_index, cot, A, top=10):
    BG = nx.Graph()
    BG.add_nodes_from(['h{}'.format(vi) for vi in cot], bipartite=0)  # hubs
    edges = [('h{}'.format(vi), int(vj)) for vi in cot for vj in np.argwhere(A[vi,:] != 0 )[:,1]]
    BG.add_nodes_from(set([e[1] for e in edges]), bipartite=1)  # authorities
    BG.add_edges_from(edges)
    centrality = Counter({n: c for n, c in nx.eigenvector_centrality_numpy(BG).items() if type(n) == int
                                                                                       and n not in cot
                                                                                       and n != node_index                                                                                    and n not in np.argwhere(A[node_index,:] != 0 )[:,1] })
    del(BG)
    #time.sleep(0.01)
    return np.asarray([n for n, pev in centrality.most_common(top)])[:top]

def frequency_by_who_to_follow(A, cot_per_node=None, p=0.85, top=10, num_cores=40):
    cot_per_node = cot_per_node if cot_per_node is not None else get_circle_of_trust_per_node(A, p, top, num_cores)
    results = Parallel(n_jobs=num_cores)(delayed(_salsa)(node_index, cot, A, top) for node_index, cot in enumerate(cot_per_node))
    unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
    del(results)
    return [0 if node_index not in unique_elements else counts_elements[np.argwhere(unique_elements == node_index)[0, 0]] for node_index in np.arange(A.shape[0])]

def who_to_follow_rank(A, njobs=1):
    return wtf_small(A, njobs)
        
def wtf_small(A, njobs):
    print('cot_per_node...')
    cot_per_node = get_circle_of_trust_per_node(A, p=0.85, top=TOPK, num_cores=njobs)

    print('cot...')
    cot = frequency_by_circle_of_trust(A, cot_per_node=cot_per_node, p=0.85, top=TOPK, num_cores=njobs)

    print('wtf...')
    wtf = frequency_by_who_to_follow(A, cot_per_node=cot_per_node, p=0.85, top=TOPK, num_cores=njobs)
    return wtf