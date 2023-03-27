
import torch
import networkx as nx
from tqdm import tqdm
import pickle

from networkx.algorithms import is_forest, is_connected
from networkx.algorithms.approximation.treewidth import treewidth_min_degree, treewidth_min_fill_in, treewidth_decomp

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from datasets.mp3d import MP3D, extract_object_graph


def analyze_objects(dset, save_file='./data/objects_analysis_all.pkl'):

    num_nodes = []
    is_tree = []
    is_disconnected = []
    tw_ub = []
    is_planar = []
    degree = []

    # breakpoint()
    len_ = len(dset)
    for idx in tqdm(range(len_), total=len_):
    # for idx in range(len_)
        data = dset[idx]

        # graph = data['dsg']
        graph_torch = data['dsg_torch']
        object_graph = extract_object_graph(graph_torch, to_nx=True)

        nodes_ = object_graph.number_of_nodes()
        edges_ = object_graph.number_of_edges()
        if nodes_:
            forest_ = is_forest(object_graph)
            disc_ = not is_connected(object_graph)
            degree_ = max([d for n, d in object_graph.degree()])
            planar_ = nx.is_planar(object_graph)
            if edges_:
                tw1, _ = treewidth_min_degree(object_graph)
                tw2, _ = treewidth_min_fill_in(object_graph)
                tw_ub.append(max([tw1, tw2]))
            else:
                tw_ub.append(0)
        else:
            forest_ = True
            disc_ = True
            degree_ = 0
            planar_ = True
            tw_ub.append(0)

        num_nodes.append(nodes_)
        is_tree.append(int(forest_))
        is_disconnected.append(int(disc_))
        degree.append(degree_)
        is_planar.append(int(planar_))

    data = dict()
    data['num_nodes'] = num_nodes
    data['is_tree'] = is_tree
    data['is_disconnected'] = is_disconnected
    data['treewidth_ub'] = tw_ub
    data['degree'] = degree
    data['is_planar'] = is_planar

    with open(save_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return None


if __name__ == "__main__":

    # usage
    # python -W ignore objects.py

    dset = MP3D(complete=True)
    analyze_objects(dset, save_file='./data/mp3d_object_analysis_complete.pkl')

    dset = MP3D(complete=False)
    analyze_objects(dset, save_file='./data/mp3d_object_analysis_trajectory.pkl')