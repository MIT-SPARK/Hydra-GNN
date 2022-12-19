import sys
from collections import Counter
import os
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import numpy as np
import torch
import torch_geometric
import spark_dsg as dsg
import glob
from pathlib import Path

from spark_dsg.mp3d import load_mp3d_info, repartition_rooms, add_gt_room_label
from spark_dsg.torch_conversion import _centroid_bbx_embedding

sys.path.append('../')
from hydra_gnn.preprocess_dsgs import convert_label_to_y, add_object_connectivity, get_room_object_dsg

BASE_DIR = './'


def extract_object_graph(graph_torch, tonx=True):
    if not graph_torch['objects']:
        object_graph = Data()
        if tonx:
            object_graph = nx.empty_graph()

    else:
        if graph_torch['objects', 'objects_to_objects', 'objects']:
            object_graph = Data(x=graph_torch['objects'].x,
                                pos=graph_torch['objects'].pos,
                                y=graph_torch['objects'].label,
                                edge_index=graph_torch['objects', 'objects_to_objects', 'objects']['edge_index'])
        else:
            object_graph = Data(x=graph_torch['objects'].x,
                                pos=graph_torch['objects'].pos,
                                y=graph_torch['objects'].label,
                                edge_index=torch.zeros(2, 0).to(dtype=torch.long))
            print("This graph has no object_to_object edges.")

        if tonx:
            object_graph = to_networkx(object_graph, to_undirected=True)
            # nx.draw(object_graph)
            # plt.show()

    return object_graph


def extract_room_graph(graph_torch, tonx=True):

    # breakpoint()
    if not graph_torch['rooms']:
        room_graph = Data()
        if tonx:
            room_graph = nx.empty_graph()

    else:
        if graph_torch['rooms', 'rooms_to_rooms', 'rooms']:
            room_graph = Data(x=graph_torch['rooms'].x,
                              pos=graph_torch['rooms'].pos,
                              y=graph_torch['rooms'].label,
                              edge_index=graph_torch['rooms', 'rooms_to_rooms', 'rooms']['edge_index'])
        else:
            room_graph = Data(x=graph_torch['rooms'].x,
                              pos=graph_torch['rooms'].pos,
                              y=graph_torch['rooms'].label,
                              edge_index=torch.zeros(2, 0).to(dtype=torch.long))

        if tonx:
            room_graph = to_networkx(room_graph, to_undirected=True)

    return room_graph


class MP3D(torch.utils.data.Dataset):
    def __init__(self, complete=True):

        if complete:
            self.hydra_dataset_dir = str(BASE_DIR) + "data/mp3d_old/hydra_mp3d_dataset/hydra_mp3d_dataset"
            self.trajectory_dirs = os.listdir(self.hydra_dataset_dir)

            self.scene_counter = Counter(full_name.split("_")[0] for full_name in self.trajectory_dirs)
            self.scene_names = list(self.scene_counter)

            self.file_names = glob.glob(self.hydra_dataset_dir + '/**/*_original_dsg.json', recursive=True)

        else:
            self.complete = complete
            self.hydra_dataset_dir = str(BASE_DIR) + "data/tro_graphs_2022_09_24"
            self.trajectory_dirs = os.listdir(self.hydra_dataset_dir)

            self.scene_counter = Counter(full_name.split("_")[0] for full_name in self.trajectory_dirs)
            self.scene_names = list(self.scene_counter)

            self.file_names = glob.glob(self.hydra_dataset_dir + '/**/gt_partial_*', recursive=True)

        self.length = len(self.file_names)
        self.file = None

    def __len__(self):

        return self.length

    def __getitem__(self, item):

        self.file = self.file_names[item]
        G = dsg.DynamicSceneGraph.load(self.file)

        dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)
        G_ro = get_room_object_dsg(G, verbose=False)
        add_object_connectivity(G_ro, threshold_near=2.0, threshold_on=1.0, max_near=2.0)

        Gtorch = G_ro.to_torch(use_heterogeneous=True)

        return {'dsg': G_ro, 'dsg_torch': Gtorch}

    def _visualize(self, G):

        dsg.render_to_open3d(G)


if __name__ == "__main__":

    from tqdm import tqdm

    dset = MP3D(complete=True)

    print(len(dset))
    for idx in tqdm(range(len(dset))):

        data = dset[idx]
        g = data['dsg']
        g_torch = data['dsg_torch']

        dset._visualize(g)




