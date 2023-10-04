"""Analyze scene graph treewidth."""
import click
import glob
import networkx as nx
import os
import pathlib
import pickle
import spark_dsg as dsg
import sys
import torch

from collections import Counter
from hydra_gnn.preprocess_dsgs import add_object_connectivity, get_room_object_dsg
from networkx.algorithms import is_forest, is_connected
from networkx.algorithms.approximation.treewidth import (
    treewidth_min_degree,
    treewidth_min_fill_in,
)
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm


def data_dir():
    """Get path to data."""
    return pathlib.Path(__file__).absolute().parent.parent / "data"


class MP3D(torch.utils.data.Dataset):
    """MP3D dataset wrapper."""

    def __init__(self, name):
        """Make a dataset."""
        self.hydra_dataset_dir = str(data_dir / name)

        self.trajectory_dirs = os.listdir(self.hydra_dataset_dir)

        self.scene_counter = Counter(
            full_name.split("_")[0] for full_name in self.trajectory_dirs
        )
        self.scene_names = list(self.scene_counter)

        self.file_names = glob.glob(
            self.hydra_dataset_dir + "/*_dsg.json", recursive=True
        )

        self.length = len(self.file_names)
        self.file = None

    def __len__(self):
        """Get the number of graphs."""
        return self.length

    def __getitem__(self, item):
        """Get a graph from the dataset."""
        self.file = self.file_names[item]
        G = dsg.DynamicSceneGraph.load(self.file)

        dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)
        G_ro = get_room_object_dsg(G, verbose=False)
        add_object_connectivity(
            G_ro, threshold_near=2.0, threshold_on=1.0, max_near=2.0
        )

        Gtorch = G_ro.to_torch(use_heterogeneous=True)

        return {"dsg": G_ro, "dsg_torch": Gtorch}

    def _visualize(self, G):
        dsg.render_to_open3d(G)


def extract_object_graph(graph_torch, tonx=True):
    """Extract object graph from torch data."""
    if not graph_torch["objects"]:
        object_graph = Data()
        if tonx:
            object_graph = nx.empty_graph()

    else:
        if graph_torch["objects", "objects_to_objects", "objects"]:
            object_graph = Data(
                x=graph_torch["objects"].x,
                pos=graph_torch["objects"].pos,
                y=graph_torch["objects"].label,
                edge_index=graph_torch["objects", "objects_to_objects", "objects"][
                    "edge_index"
                ],
            )
        else:
            object_graph = Data(
                x=graph_torch["objects"].x,
                pos=graph_torch["objects"].pos,
                y=graph_torch["objects"].label,
                edge_index=torch.zeros(2, 0).to(dtype=torch.long),
            )
            print("This graph has no object_to_object edges.")

        if tonx:
            object_graph = to_networkx(object_graph, to_undirected=True)

    return object_graph


def extract_room_graph(graph_torch, tonx=True):
    """Extract room graph from torch data."""
    if not graph_torch["rooms"]:
        room_graph = Data()
        if tonx:
            room_graph = nx.empty_graph()

    else:
        if graph_torch["rooms", "rooms_to_rooms", "rooms"]:
            room_graph = Data(
                x=graph_torch["rooms"].x,
                pos=graph_torch["rooms"].pos,
                y=graph_torch["rooms"].label,
                edge_index=graph_torch["rooms", "rooms_to_rooms", "rooms"][
                    "edge_index"
                ],
            )
        else:
            room_graph = Data(
                x=graph_torch["rooms"].x,
                pos=graph_torch["rooms"].pos,
                y=graph_torch["rooms"].label,
                edge_index=torch.zeros(2, 0).to(dtype=torch.long),
            )

        if tonx:
            room_graph = to_networkx(room_graph, to_undirected=True)

    return room_graph


def analyze_objects(dset, save_file):
    """Compute treewidth for object layer in scene graph."""
    num_nodes = []
    is_tree = []
    is_disconnected = []
    tw_ub = []
    is_planar = []
    degree = []

    len_ = len(dset)
    for idx in tqdm(range(len_), total=len_):
        data = dset[idx]
        graph_torch = data["dsg_torch"]
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
    data["num_nodes"] = num_nodes
    data["is_tree"] = is_tree
    data["is_disconnected"] = is_disconnected
    data["treewidth_ub"] = tw_ub
    data["degree"] = degree
    data["is_planar"] = is_planar

    with save_file.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return None


def analyze_rooms(dset, save_file):
    """Compute treewidth for room layer."""
    num_nodes = []
    is_tree = []
    is_disconnected = []
    tw_ub = []
    is_planar = []
    degree = []

    len_ = len(dset)
    for idx in tqdm(range(len_), total=len_):
        data = dset[idx]
        graph_torch = data["dsg_torch"]
        room_graph = extract_room_graph(graph_torch, to_nx=True)

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
    data["num_nodes"] = num_nodes
    data["is_tree"] = is_tree
    data["is_disconnected"] = is_disconnected
    data["treewidth_ub"] = tw_ub
    data["degree"] = degree
    data["is_planar"] = is_planar

    with save_file.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return None


@click.command()
@click.option(
    "-n", "--name", default="mp3d_full_backend_2023_02_19", help="dataset name"
)
@click.option("-o", "--output", default=None, help="output directory")
def main(name, output):
    """Run analysis."""
    dset = MP3D(name)
    if output:
        output_path = pathlib.Path(output).expanduser().absolute()
    else:
        output_path = data_dir / "treewidth"

    if output_path.exists():
        click.secho("Output path already exists: {output_path}!", fg="red")
        sys.exit(1)

    output_path.mkdir(parents=True)
    analyze_objects(dset, output_path / "mp3dnew_objects.pkl")
    analyze_rooms(dset, output_path / "mp3dnew_rooms.pkl")


if __name__ == "__main__":
    main()
