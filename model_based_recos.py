import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn import metrics, model_selection, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from node2vec import Node2Vec

from utils import set_seed, rewiring_list
# Hyperparameter for node2vec
DIM = 64
WALK_LEN = 10
NUM_WALKS = 200

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}

def recommender_model(G, num_cores=8):
    node2vec = Node2Vec(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores)
    model = node2vec.fit() 
    return model

def get_selected_edges(pos_edge_list, neg_edge_list):
        edges = pos_edge_list + neg_edge_list
        labels = np.zeros(len(edges))
        labels[:len(pos_edge_list)] = 1
        return edges, labels

def generate_pos_neg_links(G,seed, prop_pos=0.5, prop_neg=0.5):
        """
        Select random existing edges in the graph to be postive links,
        and random non-edges to be negative links.

        Modify graph by removing the postive links.

        prop_pos: 0.5,  # Proportion of edges to remove and use as positive samples
        prop_neg: 0.5  # Number of non-edges to use as negative samples
        """
        _rnd = np.random.RandomState(seed=seed)

        # Select n edges at random (positive samples)
        n_edges = G.number_of_edges()
        n_nodes = G.number_of_nodes()
        npos, nneg = int(prop_pos * n_edges), int(prop_neg * n_edges)
        print("Total no of edges: {}, total no of nodes:{}".format(n_edges, n_nodes))
        non_edges = [e for e in nx.non_edges(G)]
        print("Finding %d of %d non-edges" % (nneg, len(non_edges)))

        # # Select m pairs of non-edges (negative samples)
        rnd_inx = _rnd.choice(len(non_edges), nneg, replace=False)
        neg_edge_list = [non_edges[ii] for ii in rnd_inx]


        print("Finding %d positive edges of %d total edges" % (npos, n_edges))

        # # Find positive edges, and remove them.
        edges = list(G.edges())
        pos_edge_list = []
        n_count = 1
        rnd_inx = _rnd.permutation(n_edges)
        
        for eii in rnd_inx:
       
            edge = edges[eii]
            # G.remove_edge(*edge)
            pos_edge_list.append(edge)
            n_count += 1
            if n_count >= npos:
                break
        return pos_edge_list, neg_edge_list

def edges_to_features(model, edge_list, edge_function):
        n_tot = len(edge_list)
        feature_vec = np.empty((n_tot, DIM), dtype='f')
        feature_idx = np.empty((n_tot,2), dtype="i")
        for ii in range(n_tot):
            v1, v2 = edge_list[ii]

            # Edge-node features
            emb1 = np.asarray(model.wv[str(v1)])
            emb2 = np.asarray(model.wv[str(v2)])

            # Calculate edge feature
            feature_vec[ii] = edge_function(emb1, emb2)
            feature_idx[ii] = [v1,v2]
        return feature_vec, feature_idx

def train(graph, seed):

    print("Generating Node Embeddings")
    n2v_model = recommender_model(graph)

    print("Splitting Graph into Positive and Negative Edges")
    pos_edge_list, neg_edge_list = generate_pos_neg_links(graph.copy(), seed)
    edges, labels = get_selected_edges(pos_edge_list, neg_edge_list)
    print("Computing Edge Features")
    feature_vec, feature_idx = edges_to_features(n2v_model, edges, edge_functions["hadamard"])
    print("Splitting into Train Test Split")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(feature_vec, labels,feature_idx, test_size=0.33, random_state=seed)
    clf, auc_test = train_model(X_train, X_test, y_train, y_test)
    pred_edges = clf.predict(X_test)
    new_edges = idx_test[np.argwhere(pred_edges==1)]
    g_new = get_new_graph(graph.copy(), new_edges, seed)
    return g_new, auc_test


def  get_new_graph(g, new_edges, seed):
    for i, new_edge in enumerate(new_edges):
         seed += i
         set_seed(seed)
         u, v = new_edge[0][0], new_edge[0][1]
         
         if not g.has_edge(u, v):
             edges_to_be_removed = rewiring_list(g, u, 1)
             g.remove_edges_from(edges_to_be_removed) # deleting previously existing links
             g.add_edge(u,v)

    return g


def train_model(X_train, X_test, y_train, y_test):
    # Linear classifier
    scaler = StandardScaler()
    lin_clf = LogisticRegression(C=1)
    clf = pipeline.make_pipeline(scaler, lin_clf)

    # Train classifier
    clf.fit(X_train, y_train)
    auc_train = metrics._scorer.roc_auc_scorer(clf, X_train , y_train)

    # Test classifier
    auc_test = metrics._scorer.roc_auc_scorer(clf, X_test, y_test)

    print("auc train:", auc_train, "auc test", auc_test)
    return clf, auc_test



     