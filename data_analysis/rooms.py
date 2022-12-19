
import torch
import networkx as nx
from tqdm import tqdm
import pickle

from networkx.algorithms import is_forest, is_connected
from networkx.algorithms.approximation import clique
from networkx.algorithms.approximation.treewidth import treewidth_min_degree, treewidth_min_fill_in

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import sys
sys.path.append("../")
from datasets.mp3d import MP3D, extract_room_graph


def analyze_rooms(dset, save_file='./data/room_analysis_all.pkl'):

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
        room_graph = extract_room_graph(graph_torch, tonx=True)

        nodes_ = room_graph.number_of_nodes()
        edges_ = room_graph.number_of_edges()
        if nodes_:
            forest_ = is_forest(room_graph)
            disc_ = not is_connected(room_graph)
            degree_ = max([d for n, d in room_graph.degree()])
            planar_ = nx.is_planar(room_graph)
            if edges_:
                tw1, _ = treewidth_min_degree(room_graph)
                tw2, _ = treewidth_min_fill_in(room_graph)
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


def see_nontree_rooms(dset, save_dir='./data'):

    filenames_ = dict()
    filenames_["nontrees"] = []
    filenames_["nontrees_disconnected"] = []
    filenames_["nontrees_connected"] = []
    filenames_["empty"] = []

    len_ = len(dset)
    for idx in tqdm(range(len_), total=len_):
        data = dset[idx]

        # graph = data['dsg']
        graph_torch = data['dsg_torch']
        room_graph = extract_room_graph(graph_torch, tonx=True)

        nodes_ = room_graph.number_of_nodes()
        if nodes_:
            forest_ = is_forest(room_graph)
            disc_ = not is_connected(room_graph)
        else:
            forest_ = True
            disc_ = True
            filenames_["empty"].append(dset.file)

        if not forest_:
            # graph = data['dsg']
            # dset._visualize(graph)
            filenames_["nontrees"].append(dset.file)

            if disc_:
                filenames_["nontrees_disconnected"].append(dset.file)
            else:
                filenames_["nontrees_connected"].append(dset.file)

    filename = save_dir + '/rooms_that_are_not_trees.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(filenames_, f, protocol=pickle.HIGHEST_PROTOCOL)

    return None


if __name__ == "__main__":

    # usage
    # python -W ignore rooms.py

    dset = MP3D(complete=True)
    analyze_rooms(dset, save_file='./data/mp3d_room_analysis_complete.pkl')

    dset = MP3D(complete=False)
    analyze_rooms(dset, save_file='./data/mp3d_room_analysis_trajectory.pkl')
