import networkx as nx
import numpy as np
import itertools

# from IPython.display import clear_output

# Input: Graph G and number k
# output: Graph H, a subgraph of G with treewidth k


# Need to add node and link attributes
# Node attribute: score::   this will be a score function that will be
#                           updated in the algorithm.
#                           This will be initialized to 0 initially for every node
#                 weight::  this will be the weight function for each link
#                           this will be set to 0 throughout.


def graph_initialize(G):
    # Input: networkx graph G
    # Output: networkx graph G with node attribute:score and edge attribute:weight all set to zero
    score_attribute_dict = {}
    k_set_attribute_dict = {}
    weight_attribute_dict = {}
    for node in G.nodes():
        score_attribute_dict[node] = 0.0
        k_set_attribute_dict[node] = set()  # This attribute will always remain a set

    for edge in G.edges():
        weight_attribute_dict[edge] = 1.0
        # Every edge is weighed equally

    nx.set_node_attributes(G, score_attribute_dict, "score")
    nx.set_node_attributes(G, k_set_attribute_dict, "k_set")
    nx.set_edge_attributes(G, weight_attribute_dict, "weight")

    return G


def score_function(node, Clique, G, type="type2"):
    alpha = 0.1

    if type == "type1":
        # type1:: is a random score function. Eq (1) in the paper.
        return np.random.uniform(low=0.0, high=alpha, size=1)

    elif type == "type2":
        # type2:: score function in Eq (2) in the paper.
        score_temp = 0
        for neighbor in list(G.neighbors(node)):
            if neighbor in Clique:
                score_temp += G.edges[(node, neighbor)]["weight"] + np.random.uniform(
                    low=0.0, high=alpha, size=1
                )
        return score_temp

    else:
        # This means that the type is not correctly specified.
        raise ValueError


def score_update(G, k, inducted_node, inducted_to_clique):
    # Function to update score function
    # the score function needs to be updated only for nodes that are neighbors of
    # the recently inducted node!!
    #
    # Input: u:: the recently induced node (u = inducted_node)
    # Input: C:: the k-clique to which u is now attached in Ui and Ki
    #            (C = inducted_to_clique)
    # Input: G:: Graph
    # Input: k:: tree-width bound
    #
    # Step:: Loop over neighbors v of node u
    # Step:: Loop over all k-subsets S of C \union {u}
    # Step:: Compute score_function() for node=v, Clique=S, G.
    # Step:: Update score function for every neighbor v of u to this new value.
    # Return G, with updated score function.
    #
    # There is also an initial score update, which is the case when the input C is not
    # a k-clique. In this case, just update for Clique=C.
    # If C, on the other hand, has size larger than k -- Declare ValueError.

    H = G  # creating a copy of G, to be returned

    if len(inducted_to_clique) < k:
        # Case where the only set you update for is C
        S = set(inducted_to_clique)
        S.add(inducted_node)

        for v in nx.neighbors(G, inducted_node):
            H.nodes[v]["score"] = score_function(node=v, Clique=S, G=G, type="type2")
            H.nodes[v]["k_set"] = S

    elif len(inducted_to_clique) == k:
        # Case where you have to choose all k-subsets of S = C \union {u}
        S = set(inducted_to_clique)
        S.add(inducted_node)

        for v in nx.neighbors(G, inducted_node):
            # Note: This updates score and k_set values for all neighbors of v.
            #       This includes even those nodes that are already inducted.
            #       Since, we won't be using the already inducted nodes, it shouldn't matter.

            score_temp = 0.0
            set_temp = set()
            for k_set in itertools.combinations(S, k):
                # k_set is a set of size k of S = C \union {u}
                new_score = score_function(node=v, Clique=set(k_set), G=G, type="type2")
                if new_score > score_temp:
                    score_temp = new_score
                    set_temp = set(k_set)

            # Check if this assignment is possible
            H.nodes[v]["score"] = score_temp
            H.nodes[v]["k_set"] = set_temp
    else:
        raise ValueError

    return H


def key_with_max_val(d):
    """a) create a list of the dict's keys and values;
    b) return the key with the max value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def sample_node(G, taken_nodes):
    # Accessing the score values of nodes. Creates a dictionary.
    node_scores = nx.get_node_attributes(G, "score")

    # Removing all taken_nodes from the dictionary
    [node_scores.pop(key) for key in taken_nodes]

    # Collecting node with the maximum score
    sampled_node = key_with_max_val(node_scores)

    return sampled_node


def graph_add(node, node_set, Subgraph, Graph):
    # Add node to Subgraph
    # Add all links between node and node_set that exist in Graph

    ## NODE ATTRIBUTES MUST BE COPIED
    Subgraph.add_node(node)

    for v in node_set:
        if (node, v) in Graph.edges():
            # EDGE ATTRIBUTES MUST BE COPIED
            Subgraph.add_edge(node, v)

    return Subgraph


def clique_add(node, node_set, Subgraph):
    # Add node to Subgraph
    # Add all links between node and node_set

    Subgraph.add_node(node)

    for v in node_set:
        Subgraph.add_edge(node, v)

    return Subgraph


def Initialize_kTree_And_Subgraph(G, k):
    # Input G:: Graph that has valid score and weight attribute for each node and edge, respectively.
    # Input k:: Treewidth bound

    # Output U:: This is the sampled subgraph, now only initialized
    # Output K:: This is the k-tree, associated with U, now only initialized
    # Output G:: This is same as G but with updated weights
    U = nx.Graph()
    K = nx.Graph()

    # Step 1: Randomly select a node in G. Add it to U and K.
    sampled_node = np.random.choice(G.nodes())
    U.add_node(sampled_node)  # NODE ATTRIBUTES MUST BE COPIED
    K.add_node(sampled_node)
    score_update(G=G, k=k, inducted_node=sampled_node, inducted_to_clique=[])

    # Step 2: Add node u in G with maximum score. Update U and K.
    while len(U.nodes()) < k:
        taken_nodes = U.nodes()
        sampled_node = sample_node(G, taken_nodes)

        # update U
        U = graph_add(
            node=sampled_node, node_set=taken_nodes, Subgraph=U, Graph=G
        )  # NODE ATTRIBUTES MUST BE COPIED

        # update K
        K = clique_add(node=sampled_node, node_set=taken_nodes, Subgraph=K)

        # update score
        G = score_update(
            G=G, k=k, inducted_node=sampled_node, inducted_to_clique=taken_nodes
        )

    return U, K, G


def bounded_treewidth_sampling(G, k, copy_node_attributes="x", verbose=False):
    # Input: G:: NetworkX Graph
    # Input: k:: tree-width bound

    # Output: H:: Subgraph of G with tree-width <= k

    # Initialize by adding required node and edge attributes
    G = graph_initialize(G)

    # Initialize the kTree and Subgraph
    U, K, G = Initialize_kTree_And_Subgraph(G, k)

    # Loop
    progress_threshold = 0
    while len(U.nodes()) < len(G.nodes()):
        # clear_output(wait=False)
        # print("Progress Graph Sampling: ", len(U.nodes()), "/", len(G.nodes()))
        if 100.0 * len(U.nodes()) / len(G.nodes()) > progress_threshold + 5:
            progress_threshold += 5
            if verbose:
                print("Progress Graph Sampling: ", progress_threshold, "/", "100 %")

        sampled_node = sample_node(G=G, taken_nodes=U.nodes())
        C = G.nodes[sampled_node]["k_set"]

        # update U
        U = graph_add(node=sampled_node, node_set=C, Subgraph=U, Graph=G)

        # update K
        K = clique_add(node=sampled_node, node_set=C, Subgraph=K)

        # update score
        G = score_update(G=G, k=k, inducted_node=sampled_node, inducted_to_clique=C)

    if verbose:
        print("Progress Graph Sampling: 100 / 100 %")

    if copy_node_attributes is not None:
        # Copies the needed node attribute
        for node_attribute in copy_node_attributes:
            copy_node_attribute_dict = nx.get_node_attributes(G, node_attribute)
            nx.set_node_attributes(U, copy_node_attribute_dict, node_attribute)

    return U, K
